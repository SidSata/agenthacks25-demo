"""
LinguaLeap Marketing Launch Simulator – v2
Streamlit prototype featuring 3 AI agents
Run with:
    pip install streamlit pandas numpy openai python-dotenv
    streamlit run marketing_sim_app.py

Environment:
  • Set your OpenAI key in .env or Streamlit Secrets →  OPENAI_API_KEY="sk‑..."

Agents:
  1. Simulator Agent – returns CAC & sign‑ups given decisions + world event
  2. Marketing Guru Agent – suggests next‑round improvements (never perfect answer)
  3. State‑of‑World Agent – introduces random macro events each sprint
"""

import os, json, random, time
import streamlit as st
import pandas as pd
import numpy as np

# NEW: Agno and Pydantic imports
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field

# Check for agno installation
try:
    import agno # Test import
    AGNO_INSTALLED = True
except ImportError:
    AGNO_INSTALLED = False
    # This warning will be shown once Streamlit tries to render it.
    # Consider moving st.warning call to the main execution flow if needed earlier.

# --- CONFIG -----------------------------------------------------------------
WEEKS = 6
TOTAL_BUDGET = 500_000
WEEKLY_BUDGET = TOTAL_BUDGET / WEEKS

BASE_CPI = {
    "US":     {"TikTok": 3.5, "YouTube": 4.5, "Campus": 5.5},
    "India":  {"TikTok": 1.8, "YouTube": 2.5, "Campus": 3.0},
    "Brazil": {"TikTok": 2.5, "YouTube": 3.0, "Campus": 4.0},
}
CREATIVE_FACTOR = {"Basic": 1.10, "Polished": 1.00, "Viral": 0.80}
WORLD_EVENTS = [
    ("TikTok CPM Spike", 0.20),
    ("YouTube Algorithm Favors Shorts", -0.15),
    ("Campus Flu Outbreak", -0.25),
    ("Brazilian Carnival Boosts Social Reach", -0.20),
    ("US Finals Week Lowers Screen Time", 0.15),
]

# --- AGNO AGENT CONFIG & INITIALIZATION -------------------------------------
# Placed after initial imports and before it's used by agent functions.
# Streamlit elements like st.secrets and st.warning should be called where Streamlit can render them,
# typically not at the top-level during module import if they are meant to be UI warnings.
# For now, will proceed with this structure, can be refined if UI warnings are misplaced.

# This part will be defined later, near where st context is available, or handled carefully.
# For now, conceptual placement:
OPENAI_API_KEY_VALUE = None
AGNO_READY = False
simulator_agent_agno: Agent | None = None
marketing_guru_agent_agno: Agent | None = None

class SimulatorOutput(BaseModel):
    cac: float = Field(..., description="Customer Acquisition Cost, 2 decimal places.")
    signups: int = Field(..., description="Number of signups, integer.")
    explanation: str = Field(..., description="Explanation of the results (1-2 sentences).")

# Agent initialization logic will be moved into the main app flow or a setup function
# to correctly use st.secrets and st.warning.

# --- OLD OpenAI specific setup (to be removed/modified) ---
# try:
#     import openai  # Optional — graceful fallback if not present
#     OPENAI_READY = True
# except ImportError:
#     OPENAI_READY = False

# --- CONFIG (rest of it, if any, before function defs) ---
# ... existing code ...
# --- STATE MANAGEMENT --------------------------------------------------------

def init_state():
    if "week" not in st.session_state:
        st.session_state.week = 1
        st.session_state.cumulative = 0
        st.session_state.history = pd.DataFrame(
            columns=["Week", "CAC", "Signups", "Viral_K", "Event", "Guru"]
        )

def _initialize_agno_agents():
    """Helper to initialize Agno agents and manage warnings."""
    global OPENAI_API_KEY_VALUE, AGNO_READY, simulator_agent_agno, marketing_guru_agent_agno

    if not AGNO_INSTALLED:
        st.warning("`agno` package not installed. AI features will use deterministic fallback.")
        AGNO_READY = False
        return

    OPENAI_API_KEY_VALUE = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if OPENAI_API_KEY_VALUE:
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY_VALUE
        try:
            simulator_agent_agno = Agent(
                name="EdTechGrowthSimulator",
                model=OpenAIChat(id="gpt-4o"),
                instructions=[
                    "You are an ed-tech growth simulator.",
                    "Given the user's decisions (JSON format) and a world event (text description with impact factor), output a JSON with keys 'cac', 'signups', and 'explanation'.",
                    "Ensure 'cac' is a float with 2 decimal places, 'signups' is an integer.",
                    "The 'explanation' should be 1-2 sentences."
                ],
                response_model=SimulatorOutput,
            )

            marketing_guru_agent_agno = Agent(
                name="MarketingGuru",
                model=OpenAIChat(id="gpt-4o"),
                instructions=[
                    "You are a witty growth marketing guru.",
                    "NEVER give perfect answers. Instead, hint at 2-3 actionable tweaks (no more than 70 words) based on the learner's last sprint decisions and metrics (provided in JSON format).",
                    "Your reply should be plain text."
                ],
            )
            AGNO_READY = True
        except Exception as e:
            st.warning(f"Failed to initialize Agno agents: {e}. Using deterministic fallback.")
            AGNO_READY = False
    else:
        st.warning("OpenAI key not provided. AI features will use deterministic fallback.")
        AGNO_READY = False

# --- AGENTS ------------------------------------------------------------------

def state_of_world_agent():
    """Randomly pick a world event and impact multiplier (±%)."""
    event, impact = random.choice(WORLD_EVENTS)
    return {"event": event, "impact": impact}


def simulator_agent(decisions: dict, world: dict):
    """Return CAC & signups using either Agno (if available) or deterministic model."""
    mix, regions, creative, reward = (decisions[k] for k in ["mix", "regions", "creative", "reward"])
    creative_mult = CREATIVE_FACTOR[creative]
    
    spend_based_signups = 0
    for r, r_share in regions.items():
        for ch, ch_share in mix.items():
            spend = WEEKLY_BUDGET * r_share * ch_share
            # Ensure cpi calculation avoids division by zero if impact makes it -1
            impact_factor = 1 + world["impact"]
            if impact_factor <= 0: # Avoid non-positive CPI modifier
                cpi = float('inf') 
            else:
                cpi = BASE_CPI[r][ch] * creative_mult * impact_factor
            spend_based_signups += spend / cpi if cpi and cpi != float('inf') else 0
            
    viral_k = 0.20 + 0.05 * reward

    # Initialize final values with deterministic path
    final_signups = spend_based_signups * (1 + viral_k)
    final_cac = WEEKLY_BUDGET / final_signups if final_signups else float('inf') # Use inf for undefined CAC
    final_explanation = "Determined via deterministic model."

    if AGNO_READY and simulator_agent_agno:
        agent_input_str = f"Decisions: {json.dumps(decisions)}\nWorld event: {world['event']} with impact {world['impact']}"
        try:
            response: RunResponse = simulator_agent_agno.run(agent_input_str)
            if response and response.content and isinstance(response.content, SimulatorOutput):
                final_cac = response.content.cac
                final_signups = response.content.signups
                final_explanation = response.content.explanation
            else:
                st.warning("Agno Simulator Agent did not return expected output. Using deterministic results.")
        except Exception as e:
            st.warning(f"Agno Simulator Agent error – using deterministic results.\n{e}")
            
    return final_cac, final_signups, viral_k, final_explanation


def marketing_guru_agent(decisions: dict, metrics: dict):
    """Return iterative suggestions using Agno (if available) or a default message."""
    base_suggestion = "Consider tweaking your channel mix or creative intensity to balance CAC and reach."
    if not (AGNO_READY and marketing_guru_agent_agno):
        return base_suggestion

    agent_input = f"Decisions: {json.dumps(decisions)}\nMetrics: {json.dumps(metrics)}"
    try:
        response: RunResponse = marketing_guru_agent_agno.run(agent_input)
        if response and response.content and isinstance(response.content, str):
            return response.content.strip()
        else:
            st.warning("Agno Guru Agent did not return expected string output. Using default advice.")
            return base_suggestion
    except Exception as e:
        st.warning(f"Agno Guru Agent error – using default advice. {e}")
        return base_suggestion

# --- STREAMLIT UI ------------------------------------------------------------

st.set_page_config("LinguaLeap Simulator", layout="wide")
_initialize_agno_agents() # Initialize agents here where st context is available
init_state()

st.title("🎮 LinguaLeap Launch Simulator – AI Edition")

with st.sidebar:
    st.header(f"Sprint {st.session_state.week} Decisions")
    # Channel mix sliders
    tiktok = st.slider("TikTok %", 0, 100, 40)
    youtube = st.slider("YouTube %", 0, 100, 30)
    campus = 100 - tiktok - youtube
    if campus < 0:
        st.error("TikTok + YouTube cannot exceed 100%.")
        st.stop()
    st.write(f"Campus Ambassadors: **{campus}%**")

    creative = st.radio("Creative Intensity", list(CREATIVE_FACTOR.keys()), index=1)

    us = st.slider("US %", 0, 100, 34)
    india = st.slider("India %", 0, 100, 33)
    brazil = 100 - us - india
    if brazil < 0:
        st.error("US + India cannot exceed 100%.")
        st.stop()
    st.write(f"Brazil: **{brazil}%**")

    reward = st.number_input("Referral Reward $", 0.0, 5.0, 2.0, step=0.5)

    launch_btn = st.button("🚀 Launch Sprint", disabled=st.session_state.week > WEEKS)

# --- MAIN PANEL --------------------------------------------------------------

if launch_btn:
    with st.spinner("Simulating…"):
        decisions = {
            "mix": {"TikTok": tiktok/100, "YouTube": youtube/100, "Campus": campus/100},
            "regions": {"US": us/100, "India": india/100, "Brazil": brazil/100},
            "creative": creative,
            "reward": reward,
        }
        world = state_of_world_agent()
        cac, signups, viral_k, explanation = simulator_agent(decisions, world)
        st.session_state.cumulative += signups
        metrics = {"cac": cac, "signups": signups, "viral_k": viral_k}
        guru_msg = marketing_guru_agent(decisions, metrics)

        # Record history
        row = {
            "Week": st.session_state.week,
            "CAC": cac,
            "Signups": signups,
            "Viral_K": viral_k,
            "Event": world["event"],
            "Guru": guru_msg,
        }
        st.session_state.history = pd.concat(
            [st.session_state.history, pd.DataFrame([row])], ignore_index=True
        )
        st.session_state.week += 1

    st.balloons()

# --- DASHBOARD ---------------------------------------------------------------

hist = st.session_state.history
if not hist.empty:
    st.subheader("📊 Metrics")
    left, right = st.columns([2,1])
    with left:
        st.line_chart(hist.set_index("Week")["CAC"], height=250, use_container_width=True)
        st.bar_chart(hist.set_index("Week")["Signups"], height=250, use_container_width=True)
    with right:
        latest = hist.iloc[-1]
        st.metric("Latest CAC", f"${latest['CAC']:.2f}" if latest['CAC'] != float('inf') else "N/A")
        st.metric("Sign‑ups (wk)", f"{latest['Signups']}")
        st.metric("Viral K", latest['Viral_K'])
        st.write(f"**World Event:** {latest['Event']}")
        st.info(latest['Guru'])

    st.markdown("---")
    st.write("### Sprint Log")
    st.dataframe(hist, use_container_width=True, height=250)

# End‑game
if st.session_state.week > WEEKS:
    st.success("✅ Campaign complete!")
    st.metric("Average CAC", f"${hist['CAC'][hist['CAC'] != float('inf')].mean():.2f}" if not hist['CAC'][hist['CAC'] != float('inf')].empty else "N/A")
    st.metric("Total Users", int(st.session_state.cumulative))
    csv = hist.to_csv(index=False)
    st.download_button("Download run CSV", csv, "lingualleap_run_v2.csv", "text/csv")
