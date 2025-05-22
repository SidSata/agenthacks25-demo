"""
Portfolio Game Simulator – v1
A game-like interactive financial simulator for learning asset allocation, compounding, and behavioral finance.
Run with:
    pip install streamlit pandas numpy agno openai
    streamlit run portfolio_game_app.py

Environment:
  • Set your OpenAI key in .env or Streamlit Secrets →  OPENAI_API_KEY="sk‑..."

GOAL: MAKE AS MUCH MONEY AS POSSIBLE (while managing risk and life events)!
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field
import random
import matplotlib.pyplot as plt
from agno.tools.replicate import ReplicateTools
from dotenv import load_dotenv
import re
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.schema import UserMemory

load_dotenv()

# --- GAME CONSTANTS (moved up for early use) ------------------------------
AGE_START = 20
AGE_END = 30  # Only 10 years, end at age 30
YEARS = AGE_END - AGE_START
ASSET_CLASSES = ["Stocks", "Bonds", "Cash", "Alternatives"]
DEFAULT_ALLOCATION = {"Stocks": 60, "Bonds": 30, "Cash": 10, "Alternatives": 0}

# --- STREAMLIT UI -----------------------------------------------------------
st.set_page_config("\U0001F4B8 Portfolio Game Simulator", layout="wide")

# --- MODAL WORKAROUND: Show modal at the very top and block rest of app ---
def show_year_modal():
    hist = pd.DataFrame(st.session_state.history)
    if len(hist) < 1:
        return
    last = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else None
    change = last['Total'] - (prev['Total'] if prev is not None else st.session_state.starting_budget)
    change_pct = (change / (prev['Total'] if prev is not None else st.session_state.starting_budget)) * 100 if (prev is not None and prev['Total'] > 0) or (prev is None and st.session_state.starting_budget > 0) else 0
    with st.container():
        st.markdown(f"## Year {st.session_state.year} Summary")
        st.metric("Portfolio Value", f"${last['Total']:,.0f}", f"{change:+,.0f} ({change_pct:+.1f}%)")
        # Show asset growth for the year
        if prev is not None:
            growths = {}
            for k in ASSET_CLASSES:
                if prev[k] != 0:
                    growth = (last[k] - prev[k]) / prev[k] * 100
                else:
                    growth = float('nan')
                growths[k] = growth
            # Nicer display: colored badges for each asset
            badge_html = ""
            for k in ASSET_CLASSES:
                g = growths[k]
                if np.isnan(g):
                    color = '#888'
                    val = 'N/A'
                elif g > 0:
                    color = '#27ae60'
                    val = f'+{g:.1f}%'
                elif g < 0:
                    color = '#c0392b'
                    val = f'{g:.1f}%'
                else:
                    color = '#888'
                    val = '0.0%'
                badge_html += f"<span style='display:inline-block; margin-right:10px; padding:4px 12px; border-radius:12px; background:{color}; color:white; font-weight:bold;'>{k}: {val}</span>"
            st.markdown(f"**Asset Growth This Year:**<br>{badge_html}", unsafe_allow_html=True)
        else:
            st.markdown("**Asset Growth This Year:**  N/A (first year)")
        # Show the life event
        if st.session_state.event:
            st.markdown(f"### \U0001F4F0 Life Event: {st.session_state.event}")
        if st.session_state.narrative:
            st.info(st.session_state.narrative)
        if st.session_state.feedback:
            st.write(f"**Coach Dinero \U0001F9D1‍\U0001F4BC:** {st.session_state.feedback}")
        choices = st.session_state.get('coach_choices', None)
        if choices:
            st.markdown("### What will you do?")
            # Show each choice with its allocation
            choice_labels = []
            for c in choices:
                alloc_str = ', '.join([f"{k}: {c[k]}%" for k in ASSET_CLASSES])
                label = f"{c['label']}: {c['description']}\n> **New Allocation:** {alloc_str}"
                choice_labels.append(label)
            selected = st.radio("Choose your response:", choice_labels, key=f"choice_{st.session_state.year}")
            if st.button("Confirm Choice", key=f"confirm_{st.session_state.year}"):
                selected_choice = choices[choice_labels.index(selected)]
                # Set allocation in session state for next year
                for k in ASSET_CLASSES:
                    st.session_state[f'alloc_{k}'] = selected_choice[k]
                st.session_state.selected_choice = choice_labels.index(selected)
                st.session_state.show_modal = False
                st.rerun()
            st.markdown("---")
            st.markdown("Or adjust your allocation manually in the sidebar before running the next year.")
        else:
            if st.button("Continue", key=f"continue_{st.session_state.year}"):
                st.session_state.show_modal = False
                st.rerun()

if st.session_state.get('show_modal', False):
    show_year_modal()
    st.stop()

# --- AGNO AGENT SETUP -------------------------------------------------------
class CoachChoice(BaseModel):
    label: str = Field(..., description="Short label for the choice (e.g. 'Stay the Course')")
    description: str = Field(..., description="Description of the choice and its risk/reward")
    rec_stocks: float = Field(..., description="Recommended % allocation to stocks (0-100)")
    rec_bonds: float = Field(..., description="Recommended % allocation to bonds (0-100)")
    rec_cash: float = Field(..., description="Recommended % allocation to cash (0-100)")
    rec_alternatives: float = Field(..., description="Recommended % allocation to alternatives (0-100)")

class CoachFeedback(BaseModel):
    feedback: str = Field(..., description="Socratic feedback for the player")
    risky_behavior_score: float = Field(..., description="Score 0-1 for risky behavior this year")
    choices: list[CoachChoice] = Field(..., description="2-3 player choices for this event, each with a label, description, and recommended allocation.")
    rec_justification: str = Field(..., description="Justification for the recommended choices")

class NarrativeOutput(BaseModel):
    narrative: str = Field(..., description="Short immersive story for the year")

OPENAI_API_KEY_VALUE = None
REPLICATE_API_TOKEN_VALUE = None
AGNO_READY = False
coach_agent: Agent | None = None
narrative_agent: Agent | None = None

def _initialize_agno_agent():
    global OPENAI_API_KEY_VALUE, REPLICATE_API_TOKEN_VALUE, AGNO_READY, coach_agent, narrative_agent
    try:
        import agno
    except ImportError:
        st.warning("`agno` package not installed. AI features will be disabled.")
        AGNO_READY = False
        return
    OPENAI_API_KEY_VALUE = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    REPLICATE_API_TOKEN_VALUE = st.secrets.get("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_TOKEN")
    if OPENAI_API_KEY_VALUE:
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY_VALUE
    if REPLICATE_API_TOKEN_VALUE:
        if "REPLICATE_API_TOKEN" not in os.environ:
            os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN_VALUE
    if OPENAI_API_KEY_VALUE:
        try:
            coach_agent = Agent(
                name="FinanceCoach",
                model=OpenAIChat(id="gpt-4o"),
                instructions=[
                    "You are a Socratic personal finance coach. Given the player's age, asset allocation, risk buffer, recent market/life events, and their goal, give 2-line feedback and a risky-behavior score (0-1).",
                    "Always output 2-3 actionable choices for the player, each with: label, description, and a recommended allocation for the next year (stocks, bonds, cash, alternatives, each summing to 100).",
                    "Output JSON with keys: feedback, risky_behavior_score, choices (list of objects with label, description, rec_stocks, rec_bonds, rec_cash, rec_alternatives), rec_justification.",
                ],
                response_model=CoachFeedback,
            )
            narrative_agent = Agent(
                name="NarrativeAgent",
                model=OpenAIChat(id="gpt-4o"),
                instructions=[
                    "You are a creative narrator for a financial life simulation game.",
                    "Each year, given the player's age, birthplace, current phase, last event, and portfolio, write a short, immersive, second-person narrative (2-3 sentences) that sets the scene and emotional context for the year. Make it engaging and relevant to the player's situation.",
                    "Output JSON with key: narrative.",
                ],
                response_model=NarrativeOutput,
            )
            AGNO_READY = True
        except Exception as e:
            st.warning(f"Failed to initialize Agno agent: {e}. Using fallback.")
            AGNO_READY = False
    else:
        st.warning("OpenAI key not provided. AI features will be disabled.")
        AGNO_READY = False

# --- GAME CONSTANTS ---------------------------------------------------------
BIRTHPLACES = [
    ("United States", 12000, 60000),
    ("India", 3000, 15000),
    ("Brazil", 4000, 20000),
    ("Germany", 8000, 40000),
    ("China", 5000, 25000),
    ("Nigeria", 2000, 8000),
    ("Japan", 9000, 35000),
    ("United Kingdom", 10000, 42000),
    ("South Africa", 3500, 12000),
    ("Mexico", 3500, 14000),
]

# Historical/synthetic returns (annualized, lognormal params)
RETURNS = {
    "Stocks": (0.07, 0.16),
    "Bonds": (0.03, 0.06),
    "Cash": (0.01, 0.01),
    "Alternatives": (0.09, 0.25),
}
INFLATION_MEAN, INFLATION_STD = 0.025, 0.01
SALARY_GROWTH_MEAN, SALARY_GROWTH_STD = 0.03, 0.02

# --- RANDOM EVENTS (WITH CHOICES) -------------------------------------------
NEWS_EVENTS = [
    # --- Normal (60%) ---
    ("No major events this year.", {}, None),
    ("Steady economic growth. Markets are calm.", {}, None),
    ("Mild inflation. Cash loses 2% value.", {"Cash": -0.02}, None),
    ("Interest rates unchanged. Bonds steady.", {"Bonds": 0.0}, None),
    ("Stable job market. No major changes.", {}, None),
    ("Mild market volatility. Small ups and downs.", {}, None),
    ("Slight uptick in tech stocks. Stocks up 3%.", {"Stocks": 0.03}, None),
    ("Healthcare sector stable. No effect.", {}, None),
    ("Consumer confidence steady. No effect.", {}, None),
    ("Mild currency fluctuation. Cash up 1%.", {"Cash": 0.01}, None),
    ("Bond market quiet. No effect.", {}, None),
    ("Alternatives see mild growth. Alternatives up 2%.", {"Alternatives": 0.02}, None),
    # --- Great (20%) ---
    ("Bull market! Stocks up 15%.", {"Stocks": 0.15}, None),
    ("Bond rally! Bonds up 10%.", {"Bonds": 0.1}, None),
    ("Crypto boom! Alternatives up 20%.", {"Alternatives": 0.2}, None),
    ("Unexpected inheritance! Cash up 20%.", {"Cash": 0.2}, None),
    ("Tech breakthrough! Stocks up 20%.", {"Stocks": 0.2}, None),
    ("Salary bonus! Cash up 10%.", {"Cash": 0.1}, None),
    ("Emerging markets surge! Alternatives up 15%.", {"Alternatives": 0.15}, None),
    # --- Bad (20%) ---
    ("Dot-com crash! Stocks drop 30%.", {"Stocks": -0.3}, None),
    ("Tech Bubble Burst! Your tech-heavy stock allocation has taken a 40% hit.",
     {"Stocks": -0.4},
     [
         {"label": "Stay the Course", "description": "Ride it out, hope for recovery. (Risk: Further drops)"},
         {"label": "Rebalance Now", "description": "Sell some of what's left of stocks and buy more bonds/cash. (Risk: Miss out on a quick rebound, lock in some loss)"},
         {"label": "Double Down", "description": "Buy more stocks at these lower prices. (Risk: High risk, high potential reward/loss)"},
     ]),
    ("Layoff! Must withdraw 6 months' salary.", "layoff", None),
    ("Medical bill! Emergency fund needed.", "medical", None),
    ("Inflation spike! Cash loses 5% value.", {"Cash": -0.05}, None),
    ("Bond market crash! Bonds down 12%.", {"Bonds": -0.12}, None),
    ("Crypto winter! Alternatives down 18%.", {"Alternatives": -0.18}, None),
    ("Unexpected tax bill! Cash down 8%.", {"Cash": -0.08}, None),
    ("Natural disaster! Stocks and bonds both down 7%.", {"Stocks": -0.07, "Bonds": -0.07}, None),
]

# --- SESSION STATE INIT -----------------------------------------------------
def init_state():
    if "year" not in st.session_state:
        birthplace, min_budget, max_budget = random.choice(BIRTHPLACES)
        starting_budget = random.randint(min_budget, max_budget)
        st.session_state.birthplace = birthplace
        st.session_state.starting_budget = starting_budget
        st.session_state.year = 0
        st.session_state.age = AGE_START
        st.session_state.salary = random.randint(20000, 80000)
        st.session_state.portfolio = {k: starting_budget * v / 100 for k, v in DEFAULT_ALLOCATION.items()}
        st.session_state.history = []
        st.session_state.snowball = 0
        st.session_state.risk_score = 0
        st.session_state.event = ""
        st.session_state.emergency_fund = 6 # months
        st.session_state.rebalance = True
        st.session_state.contribution_pct = 10
        st.session_state.withdrawal = 0
        st.session_state.phase = "Early-career"
        st.session_state.feedback = ""
        st.session_state.risky_behavior_score = 0
        st.session_state.narrative = ""
        st.session_state.last_coach_advice = None
        st.session_state.ignored_advice_count = 0
        st.session_state.player_outperformed_count = 0

# --- GAME PHASES ------------------------------------------------------------
def get_phase(age):
    if age < 31:
        return "Early-career"
    elif age < 51:
        return "Mid-career"
    elif age < 66:
        return "Pre-retirement"
    else:
        return "Retirement"

# --- SIMULATION ENGINE (MODIFIED TO HANDLE CHOICES) -------------------------
def run_year(allocation, contribution_pct, rebalance, risk_buffer, withdrawal, player_choice=None):
    # Salary growth
    salary = st.session_state.salary * np.random.normal(1 + SALARY_GROWTH_MEAN, SALARY_GROWTH_STD)
    # Inflation
    inflation = np.random.normal(INFLATION_MEAN, INFLATION_STD)
    # Market returns
    returns = {k: np.random.normal(RETURNS[k][0], RETURNS[k][1]) for k in ASSET_CLASSES}
    # News event
    event, effect, choices = random.choice(NEWS_EVENTS)
    st.session_state.current_event_choices = choices
    st.session_state.current_event_label = event
    st.session_state.current_event_effect = effect
    # Apply event
    if isinstance(effect, dict):
        for k, v in effect.items():
            returns[k] += v
    elif effect == "layoff":
        withdrawal += salary / 2 # 6 months' salary
        event += f" Forced withdrawal: ${withdrawal:,.0f}."
    elif effect == "medical":
        withdrawal += salary / 4 # 3 months' salary
        event += f" Medical bill: ${withdrawal:,.0f}."
    # If player_choice is given and choices exist, apply additional logic (placeholder)
    if choices and player_choice is not None:
        # Example: modify returns or allocations based on choice
        if player_choice == 0:  # Stay the Course
            pass  # No change
        elif player_choice == 1:  # Rebalance Now
            # Move 20% from stocks to bonds/cash
            move = min(st.session_state.portfolio["Stocks"] * 0.2, st.session_state.portfolio["Stocks"])
            st.session_state.portfolio["Stocks"] -= move
            st.session_state.portfolio["Bonds"] += move / 2
            st.session_state.portfolio["Cash"] += move / 2
        elif player_choice == 2:  # Double Down
            # Add 10% more to stocks from cash if possible
            move = min(st.session_state.portfolio["Cash"] * 0.1, st.session_state.portfolio["Cash"])
            st.session_state.portfolio["Cash"] -= move
            st.session_state.portfolio["Stocks"] += move
    # Asset growth
    portfolio = st.session_state.portfolio.copy()
    total = sum(portfolio.values())
    # Contribution
    contrib = salary * contribution_pct / 100
    for k in ASSET_CLASSES:
        portfolio[k] += contrib * allocation[k] / 100
    # Withdrawals (from cash first, then bonds, then stocks, then alternatives)
    w = withdrawal
    for k in ["Cash", "Bonds", "Stocks", "Alternatives"]:
        take = min(portfolio[k], w)
        portfolio[k] -= take
        w -= take
        if w <= 0:
            break
    # Apply returns
    for k in ASSET_CLASSES:
        portfolio[k] *= np.exp(returns[k])
    # Inflation adjustment
    for k in ASSET_CLASSES:
        portfolio[k] /= (1 + inflation)
    # Rebalance
    if rebalance:
        total = sum(portfolio.values())
        for k in ASSET_CLASSES:
            portfolio[k] = total * allocation[k] / 100
    # Snowball meter: sum of all investment growth over time
    snowball = st.session_state.snowball + sum(portfolio.values()) - sum(st.session_state.portfolio.values())
    # Risk barometer: worst 5% year (simulate 1000 draws)
    sim_returns = [sum([np.random.normal(RETURNS[k][0], RETURNS[k][1]) * allocation[k] / 100 for k in ASSET_CLASSES]) for _ in range(1000)]
    var_5 = np.percentile(sim_returns, 5)
    # Update state
    st.session_state.salary = salary
    st.session_state.portfolio = portfolio
    st.session_state.snowball = snowball
    st.session_state.risk_score = var_5
    st.session_state.event = event
    st.session_state.age += 1
    st.session_state.year += 1
    st.session_state.phase = get_phase(st.session_state.age)
    # Record history
    st.session_state.history.append({
        'Year': st.session_state.year,
        'Age': st.session_state.age,
        'Phase': st.session_state.phase,
        'Salary': salary,
        **portfolio,
        'Total': sum(portfolio.values()),
        'Contribution': contrib,
        'Withdrawal': withdrawal,
        'Event': event,
        'Snowball': snowball,
        'VaR_5': var_5,
    })

# --- STREAMLIT UI -----------------------------------------------------------
_initialize_agno_agent()
init_state()

st.title("💸 Portfolio Game Simulator")
st.markdown(f"**You were born in {st.session_state.birthplace} with a starting budget of ${st.session_state.starting_budget:,}.**")

# --- SIDEBAR: Controls ---
with st.sidebar:
    # Apply coach recommendation to sliders if requested
    if st.session_state.get('apply_coach', False) and st.session_state.get('coach_recommendation'):
        for k, v in zip(ASSET_CLASSES, [
            st.session_state['coach_recommendation']['Stocks'],
            st.session_state['coach_recommendation']['Bonds'],
            st.session_state['coach_recommendation']['Cash'],
            st.session_state['coach_recommendation']['Alternatives'],
        ]):
            st.session_state[f'alloc_{k}'] = int(v)
        st.session_state['apply_coach'] = False
        st.rerun()
    st.header("Your Yearly Plan")
    st.write(f"**Year {st.session_state.year+1}  |  Age {st.session_state.age}**")
    # User-settable goal
    goal = st.number_input("Set your target portfolio value at age 30 ($)", 10000, 10000000, 1000000, step=10000)
    st.session_state.goal = goal
    # Dynamic allocation sliders that cannot sum above 100%
    alloc = {}
    remaining = 100
    for i, k in enumerate(ASSET_CLASSES):
        total_portfolio = sum(st.session_state.portfolio.values())
        if st.session_state.year == 0 or total_portfolio == 0:
            default = int(DEFAULT_ALLOCATION[k])
        else:
            default = int(st.session_state.portfolio[k] / total_portfolio * 100)
        slider_key = f'alloc_{k}'
        if slider_key not in st.session_state:
            st.session_state[slider_key] = min(default, remaining)
        max_val = remaining if i == len(ASSET_CLASSES)-1 else remaining
        if max_val > 0:
            alloc[k] = st.slider(f"{k} %", 0, max_val, key=slider_key)
        else:
            alloc[k] = 0
        remaining -= alloc[k]
    if sum(alloc.values()) != 100:
        st.warning("Asset allocation must sum to 100%.")
        st.stop()
    # --- Dynamic Risk Score Calculation ---
    # Use the same logic as in run_year for VaR_5 (5th percentile worst year)
    RETURNS = {
        "Stocks": (0.07, 0.16),
        "Bonds": (0.03, 0.06),
        "Cash": (0.01, 0.01),
        "Alternatives": (0.09, 0.25),
    }
    sim_returns = [sum([np.random.normal(RETURNS[k][0], RETURNS[k][1]) * alloc[k] / 100 for k in ASSET_CLASSES]) for _ in range(1000)]
    var_5 = np.percentile(sim_returns, 5)
    # Color code: green (>-0.05), orange (-0.10 to -0.05), red (<-0.10)
    if var_5 > -0.05:
        risk_color = 'green'
    elif var_5 > -0.10:
        risk_color = 'orange'
    else:
        risk_color = 'red'
    st.markdown(f"<span style='font-size:1.2em; font-weight:bold;'>Risk Score (VaR 5%): <span style='color:{risk_color};'>{var_5:.1%}</span></span>", unsafe_allow_html=True)
    st.markdown("---")
    # Show current mix pie chart in sidebar
    pf = st.session_state.portfolio
    if sum(pf.values()) > 0:
        fig, ax = plt.subplots()
        ax.pie(list(pf.values()), labels=list(pf.keys()), autopct='%1.0f%%', startangle=90)
        ax.set_title('Current Mix')
        st.pyplot(fig)
    else:
        st.info("No assets in portfolio to display mix.")
    st.markdown("---")
    rebalance = st.checkbox("Auto-rebalance portfolio", value=st.session_state.rebalance)
    st.markdown("---")
    st.caption("Adjust your plan, then click **Run Year** at the top right.")

# Set these to session state for simulation logic (not user-editable)
contribution_pct = st.session_state.contribution_pct
risk_buffer = st.session_state.emergency_fund
withdrawal = st.session_state.withdrawal

# --- MEMORY SETUP -----------------------------------------------------------
os.makedirs('tmp', exist_ok=True)
memory_db = SqliteMemoryDb(
    table_name="user_memories",
    db_file="tmp/memory.db"
)
memory = Memory(db=memory_db)
user_id = "player1"  # You can make this dynamic if you want

# --- MAIN LAYOUT ---
main_left, main_right = st.columns([3, 2], gap="large")

with main_left:
    # Portfolio value very prominent
    pf = st.session_state.portfolio
    st.markdown("# 💰 Portfolio Value")
    st.markdown(f"<h1 style='font-size:3em; color:green;'>${sum(pf.values()):,.0f}</h1>", unsafe_allow_html=True)
    st.markdown(f"<b>Goal for age 30:</b> ${st.session_state.goal:,.0f}", unsafe_allow_html=True)
    # Narrative at the top
    if st.session_state.narrative:
        st.info(st.session_state.narrative)
    st.markdown("---")
    st.subheader("Current Portfolio")
    st.metric("Salary", f"${st.session_state.salary:,.0f}")
    st.metric("Phase", st.session_state.phase)
    st.markdown("---")
    # Portfolio value chart
    hist = pd.DataFrame(st.session_state.history)
    if not hist.empty:
        st.subheader("📈 Portfolio Value Over Time")
        st.line_chart(hist.set_index('Year')[['Total']], use_container_width=True, height=300)
        st.subheader("📊 Asset Mix by Year")
        mix_by_year = hist.groupby('Year')[ASSET_CLASSES].mean()
        st.bar_chart(mix_by_year, use_container_width=True, height=200)
        # --- Asset Growth by Year ---
        st.subheader("📈 Asset Growth by Year")
        # Calculate yearly percentage growth for Stocks, Bonds, Cash
        growth_df = hist[['Year', 'Stocks', 'Bonds', 'Cash']].copy()
        growth_df.set_index('Year', inplace=True)
        growth_pct = growth_df.pct_change().fillna(0) * 100
        st.line_chart(growth_pct, use_container_width=True, height=200)
        st.markdown("---")
        st.write("### Year-by-Year Breakdown")
        st.dataframe(hist, use_container_width=True, height=350)
    # Endgame
    if st.session_state.age >= AGE_END:
        st.success("🏁 Game Over! Here's your final score:")
        final = hist.iloc[-1]
        st.metric("Final Portfolio Value", f"${final['Total']:,.0f}")
        st.metric("Max Portfolio Value", f"${hist['Total'].max():,.0f}")
        st.metric("Goal Achieved", "Yes" if final['Total'] >= st.session_state.goal else "No")
        if (hist['Withdrawal'] == 0).all() and (hist['Cash'] >= final['Salary']/2).all():
            st.balloons()
            st.success("You earned the Financial Resilience badge!")
        st.markdown("---")
        st.write("#### Reflection Worksheet")
        st.write("What single decision most improved your outcome?")
        st.text_area("Your answer:")
        st.write("Download your run:")
        csv = hist.to_csv(index=False)
        st.download_button("Download CSV", csv, "portfolio_game_run.csv", "text/csv")

with main_right:
    # Prominent Run Year button at the top right using Streamlit's native button
    run_year_btn = st.button("🚀 Run Year", key="run_year_top", use_container_width=True)
    simulate_5_btn = st.button("⏩ Simulate 5 Years", key="simulate_5_years", use_container_width=True)
    simulate_end_btn = st.button("🏁 Simulate Till End", key="simulate_till_end", use_container_width=True)
    # Sticky news feed and AI coach advice
    with st.container():
        st.markdown("### 📰 News Feed & AI Coach")
        st.write(f"**Last Event:** {st.session_state.event}")
        st.write(f"**Coach Dinero 🧑‍💼:** {st.session_state.feedback}")
        # Show coach's recommended allocation if available
        rec = st.session_state.get('coach_recommendation', None)
        if rec:
            st.markdown("**Coach's Recommended Allocation:**")
            st.write(f"- Stocks: {rec['Stocks']}%\n- Bonds: {rec['Bonds']}%\n- Cash: {rec['Cash']}%\n- Alternatives: {rec['Alternatives']}%")
            if rec.get('Justification'):
                st.info(f"**Coach's Justification:** {rec['Justification']}")
            if st.button("Apply Coach Recommendation", key="apply_coach_alloc"):
                st.session_state['apply_coach'] = True
                st.rerun()
        # --- Yearly return and grade ---
        hist = pd.DataFrame(st.session_state.history)
        if len(hist) > 1:
            last = hist.iloc[-1]
            prev = hist.iloc[-2]
            start_val = prev['Total']
            end_val = last['Total']
            if start_val > 0:
                pct_return = (end_val - start_val) / start_val * 100
            else:
                pct_return = 0
            # Letter grade logic
            if pct_return >= 10:
                grade = 'A'
                grade_color = 'green'
            elif pct_return >= 5:
                grade = 'B'
                grade_color = 'blue'
            elif pct_return >= 0:
                grade = 'C'
                grade_color = 'orange'
            elif pct_return >= -5:
                grade = 'D'
                grade_color = 'red'
            else:
                grade = 'F'
                grade_color = 'darkred'
            st.markdown(f"**Yearly Return:** {pct_return:+.1f}%")
            st.markdown(f"<span style='font-size:2em; font-weight:bold; color:{grade_color};'>Grade: {grade}</span>", unsafe_allow_html=True)
        else:
            st.markdown("**Yearly Return:** N/A (Grade: N/A)")

# --- Run Year Button logic ---
if run_year_btn:
    with st.spinner("Simulating year and getting coach advice..."):
        # Get last year's return for memory
        last_yearly_return = 0
        if len(st.session_state.history) > 0:
            hist = pd.DataFrame(st.session_state.history)
            if len(hist) > 1:
                last_yearly_return = (hist.iloc[-1]['Total'] - hist.iloc[-2]['Total']) / hist.iloc[-2]['Total'] * 100 if hist.iloc[-2]['Total'] > 0 else 0
            else:
                last_yearly_return = (hist.iloc[-1]['Total'] - st.session_state.starting_budget) / st.session_state.starting_budget * 100 if st.session_state.starting_budget > 0 else 0

        # Add a memory of last year's action and outcome
        ignored_advice = False
        if st.session_state.last_coach_advice:
            # Simple check if allocation was different
            if any(alloc[k] != st.session_state.last_coach_advice[k] for k in ASSET_CLASSES):
                ignored_advice = True
                st.session_state.ignored_advice_count += 1
            if last_yearly_return > 5 and ignored_advice: # Arbitrary threshold for outperformance
                st.session_state.player_outperformed_count += 1

        action_summary = f"Year {st.session_state.year}: Player allocated: {alloc}. "
        if st.session_state.last_coach_advice:
            action_summary += f"Coach recommended: {st.session_state.last_coach_advice}. "
        if ignored_advice:
            action_summary += "(Player ignored advice). "
        action_summary += f"Outcome: {last_yearly_return:.1f}% return."
        memory.add_user_memory(
            memory=UserMemory(
                memory=action_summary,
                topics=["actions", "advice", "outcome"]
            ),
            user_id=user_id
        )

        run_year(alloc, contribution_pct, rebalance, risk_buffer, withdrawal)
        # AI Coach feedback
        feedback = ""
        risky_behavior_score = 0
        narrative = ""
        coach_choices = None
        coach_justification = None
        if AGNO_READY and coach_agent:
            # Retrieve memories for the coach
            recent_memories = memory.get_user_memories(user_id=user_id)
            memories_text = "\n".join([m.memory for m in recent_memories[-5:]])  # Last 5 years
            ignored_count = st.session_state.ignored_advice_count
            outperformed_count = st.session_state.player_outperformed_count
            prompt = f"""You are Coach Dinero 🧑‍💼, a Socratic personal finance coach. You can recall the player's past actions. Player has ignored your advice {ignored_count} times and outperformed you {outperformed_count} times.

Player's Past Actions (last 5 years):
{memories_text}

Current Situation:
Age: {st.session_state.age}
Current Allocation: {alloc}
Risk Buffer: {risk_buffer}
Last Market Event: {st.session_state.event}
Player's Goal (at age 30): ${st.session_state.goal:,.0f}

Give 2-line feedback and a risky-behavior score (0-1). Always output 2-3 actionable choices for the player, each with: label, description, and a recommended allocation for the next year (stocks, bonds, cash, alternatives, each summing to 100).
"""
            try:
                response: RunResponse = coach_agent.run(prompt)
                if response and response.content and isinstance(response.content, CoachFeedback):
                    feedback = response.content.feedback
                    risky_behavior_score = response.content.risky_behavior_score
                    coach_choices = [
                        {
                            'label': c.label,
                            'description': c.description,
                            'Stocks': int(c.rec_stocks),
                            'Bonds': int(c.rec_bonds),
                            'Cash': int(c.rec_cash),
                            'Alternatives': int(c.rec_alternatives),
                        } for c in response.content.choices
                    ]
                    coach_justification = response.content.rec_justification
                    # Default to first choice for last_coach_advice
                    if coach_choices:
                        st.session_state.last_coach_advice = {k: coach_choices[0][k] for k in ASSET_CLASSES}
            except Exception as e:
                feedback = f"AI coach error: {e}"
        if AGNO_READY and narrative_agent: # Check if narrative_agent is not None
            n_prompt = f"Age: {st.session_state.age}\nBirthplace: {st.session_state.birthplace}\nPhase: {st.session_state.phase}\nEvent: {st.session_state.event}\nPortfolio: {st.session_state.portfolio}"
            try:
                n_response: RunResponse = narrative_agent.run(n_prompt)
                if n_response and n_response.content and isinstance(n_response.content, NarrativeOutput):
                    narrative = n_response.content.narrative
            except Exception as e:
                narrative = f"Narrative agent error: {e}"
        st.session_state.feedback = feedback
        st.session_state.risky_behavior_score = risky_behavior_score
        st.session_state.narrative = narrative
        st.session_state.coach_choices = coach_choices
        st.session_state.coach_justification = coach_justification
        if st.session_state.age < AGE_END:
            st.session_state.show_modal = True
        st.rerun()

# --- Simulate 5 Years Button logic ---
if simulate_5_btn:
    with st.spinner("Simulating 5 years..."):
        years_to_sim = min(5, AGE_END - st.session_state.age + 1)
        for _ in range(years_to_sim):
            if st.session_state.age > AGE_END or st.session_state.get('show_modal', False):
                break
            # Repeat the same logic as run_year_btn, but skip modal until needed
            last_yearly_return = 0
            if len(st.session_state.history) > 0:
                hist = pd.DataFrame(st.session_state.history)
                if len(hist) > 1:
                    last_yearly_return = (hist.iloc[-1]['Total'] - hist.iloc[-2]['Total']) / hist.iloc[-2]['Total'] * 100 if hist.iloc[-2]['Total'] > 0 else 0
                else:
                    last_yearly_return = (hist.iloc[-1]['Total'] - st.session_state.starting_budget) / st.session_state.starting_budget * 100 if st.session_state.starting_budget > 0 else 0
            ignored_advice = False
            if st.session_state.last_coach_advice:
                if any(alloc[k] != st.session_state.last_coach_advice[k] for k in ASSET_CLASSES):
                    ignored_advice = True
                    st.session_state.ignored_advice_count += 1
                if last_yearly_return > 5 and ignored_advice:
                    st.session_state.player_outperformed_count += 1
            action_summary = f"Year {st.session_state.year}: Player allocated: {alloc}. "
            if st.session_state.last_coach_advice:
                action_summary += f"Coach recommended: {st.session_state.last_coach_advice}. "
            if ignored_advice:
                action_summary += "(Player ignored advice). "
            action_summary += f"Outcome: {last_yearly_return:.1f}% return."
            memory.add_user_memory(
                memory=UserMemory(
                    memory=action_summary,
                    topics=["actions", "advice", "outcome"]
                ),
                user_id=user_id
            )
            run_year(alloc, contribution_pct, rebalance, risk_buffer, withdrawal)
            # If a modal is triggered (e.g., for a life event/choice), break and show it
            if st.session_state.get('show_modal', False):
                break
        if st.session_state.age < AGE_END:
            st.session_state.show_modal = True
        st.rerun()

# --- Simulate Till End Button logic ---
if simulate_end_btn:
    with st.spinner("Simulating till end of game..."):
        while st.session_state.age <= AGE_END and not st.session_state.get('show_modal', False):
            last_yearly_return = 0
            if len(st.session_state.history) > 0:
                hist = pd.DataFrame(st.session_state.history)
                if len(hist) > 1:
                    last_yearly_return = (hist.iloc[-1]['Total'] - hist.iloc[-2]['Total']) / hist.iloc[-2]['Total'] * 100 if hist.iloc[-2]['Total'] > 0 else 0
                else:
                    last_yearly_return = (hist.iloc[-1]['Total'] - st.session_state.starting_budget) / st.session_state.starting_budget * 100 if st.session_state.starting_budget > 0 else 0
            ignored_advice = False
            if st.session_state.last_coach_advice:
                if any(alloc[k] != st.session_state.last_coach_advice[k] for k in ASSET_CLASSES):
                    ignored_advice = True
                    st.session_state.ignored_advice_count += 1
                if last_yearly_return > 5 and ignored_advice:
                    st.session_state.player_outperformed_count += 1
            action_summary = f"Year {st.session_state.year}: Player allocated: {alloc}. "
            if st.session_state.last_coach_advice:
                action_summary += f"Coach recommended: {st.session_state.last_coach_advice}. "
            if ignored_advice:
                action_summary += "(Player ignored advice). "
            action_summary += f"Outcome: {last_yearly_return:.1f}% return."
            memory.add_user_memory(
                memory=UserMemory(
                    memory=action_summary,
                    topics=["actions", "advice", "outcome"]
                ),
                user_id=user_id
            )
            run_year(alloc, contribution_pct, rebalance, risk_buffer, withdrawal)
            if st.session_state.get('show_modal', False):
                break
        if st.session_state.age < AGE_END:
            st.session_state.show_modal = True
        st.rerun() 