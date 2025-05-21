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

# --- AGNO AGENT SETUP -------------------------------------------------------
class CoachFeedback(BaseModel):
    feedback: str = Field(..., description="Socratic feedback for the player")
    risky_behavior_score: float = Field(..., description="Score 0-1 for risky behavior this year")
    rec_stocks: float = Field(..., description="Recommended % allocation to stocks (0-100)")
    rec_bonds: float = Field(..., description="Recommended % allocation to bonds (0-100)")
    rec_cash: float = Field(..., description="Recommended % allocation to cash (0-100)")
    rec_alternatives: float = Field(..., description="Recommended % allocation to alternatives (0-100)")

class NarrativeOutput(BaseModel):
    narrative: str = Field(..., description="Short immersive story for the year")

# For video narration: more scripted, cinematic
class VideoNarrationOutput(BaseModel):
    script: str = Field(..., description="A short, cinematic narration script for the year")

OPENAI_API_KEY_VALUE = None
AGNO_READY = False
coach_agent: Agent | None = None
narrative_agent: Agent | None = None

def _initialize_agno_agent():
    global OPENAI_API_KEY_VALUE, AGNO_READY, coach_agent, narrative_agent
    try:
        import agno
    except ImportError:
        st.warning("`agno` package not installed. AI features will be disabled.")
        AGNO_READY = False
        return
    OPENAI_API_KEY_VALUE = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY_VALUE:
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY_VALUE
        try:
            coach_agent = Agent(
                name="FinanceCoach",
                model=OpenAIChat(id="gpt-4o"),
                instructions=[
                    "You are a Socratic personal finance coach. Given the player's age, asset allocation, risk buffer, recent market/life events, and their goal, give 2-line feedback and a risky-behavior score (0-1).",
                    "Always output a recommended allocation for the next year as four numbers (stocks, bonds, cash, alternatives) that sum to 100. Output JSON with keys: feedback, risky_behavior_score, rec_stocks, rec_bonds, rec_cash, rec_alternatives.",
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
AGE_START = 20
AGE_END = 70
YEARS = AGE_END - AGE_START
ASSET_CLASSES = ["Stocks", "Bonds", "Cash", "Alternatives"]
DEFAULT_ALLOCATION = {"Stocks": 60, "Bonds": 30, "Cash": 10, "Alternatives": 0}
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

# --- RANDOM EVENTS ----------------------------------------------------------
NEWS_EVENTS = [
    ("Dot-com crash! Stocks drop 30%.", {"Stocks": -0.3}),
    ("Bond rally! Bonds up 10%.", {"Bonds": 0.1}),
    ("Crypto boom! Alternatives up 20%.", {"Alternatives": 0.2}),
    ("Inflation spike! Cash loses 5% value.", {"Cash": -0.05}),
    ("Layoff! Must withdraw 6 months' salary.", "layoff"),
    ("Medical bill! Emergency fund needed.", "medical"),
    ("Bull market! Stocks up 15%.", {"Stocks": 0.15}),
    ("No major events this year.", {}),
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

# --- SIMULATION ENGINE ------------------------------------------------------
def run_year(allocation, contribution_pct, rebalance, risk_buffer, withdrawal):
    # Salary growth
    salary = st.session_state.salary * np.random.normal(1 + SALARY_GROWTH_MEAN, SALARY_GROWTH_STD)
    # Inflation
    inflation = np.random.normal(INFLATION_MEAN, INFLATION_STD)
    # Market returns
    returns = {k: np.random.normal(RETURNS[k][0], RETURNS[k][1]) for k in ASSET_CLASSES}
    # News event
    event, effect = random.choice(NEWS_EVENTS)
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
st.set_page_config("💸 Portfolio Game Simulator", layout="wide")
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
    goal = st.number_input("Set your target portfolio value at age 70 ($)", 10000, 10000000, 1000000, step=10000)
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

# --- MAIN LAYOUT ---
main_left, main_right = st.columns([3, 2], gap="large")

with main_left:
    # Portfolio value very prominent
    pf = st.session_state.portfolio
    st.markdown("# 💰 Portfolio Value")
    st.markdown(f"<h1 style='font-size:3em; color:green;'>${sum(pf.values()):,.0f}</h1>", unsafe_allow_html=True)
    st.markdown(f"<b>Goal for age 70:</b> ${st.session_state.goal:,.0f}", unsafe_allow_html=True)
    # Narrative at the top
    if st.session_state.narrative:
        st.info(st.session_state.narrative)
    # Show video narration if available
    st.markdown("""
    **Goal:** Make as much money as possible by age 70—while managing risk, life events, and your own behavior!
    """)
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
        st.subheader("📊 Asset Mix by Decade")
        decade = (hist['Age']//10)*10
        mix_by_decade = hist.groupby(decade)[ASSET_CLASSES].mean()
        st.bar_chart(mix_by_decade, use_container_width=True, height=200)
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
    # Sticky news feed and AI coach advice
    with st.container():
        st.markdown("### 📰 News Feed & AI Coach")
        st.write(f"**Last Event:** {st.session_state.event}")
        st.write(f"**Coach:** {st.session_state.feedback}")
        # Show coach's recommended allocation if available
        rec = st.session_state.get('coach_recommendation', None)
        if rec:
            st.markdown("**Coach's Recommended Allocation:**")
            st.write(f"- Stocks: {rec['Stocks']}%\n- Bonds: {rec['Bonds']}%\n- Cash: {rec['Cash']}%\n- Alternatives: {rec['Alternatives']}%")
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
            elif pct_return >= 5:
                grade = 'B'
            elif pct_return >= 0:
                grade = 'C'
            elif pct_return >= -5:
                grade = 'D'
            else:
                grade = 'F'
            st.markdown(f"**Yearly Return:** {pct_return:+.1f}% (Grade: {grade})")
            st.progress(min(1, max(0, pct_return/20)))
        else:
            st.markdown("**Yearly Return:** N/A (Grade: N/A)")
            st.progress(0)

# --- Run Year Button logic ---
if run_year_btn:
    run_year(alloc, contribution_pct, rebalance, risk_buffer, withdrawal)
    # AI Coach feedback
    feedback = ""
    risky_behavior_score = 0
    narrative = ""
    coach_recommendation = None
    if AGNO_READY and coach_agent:
        prompt = f"Age: {st.session_state.age}\nAllocation: {alloc}\nRisk buffer: {risk_buffer}\nEvent: {st.session_state.event}\nGoal: {st.session_state.goal}"
        try:
            response: RunResponse = coach_agent.run(prompt)
            if response and response.content and isinstance(response.content, CoachFeedback):
                feedback = response.content.feedback
                risky_behavior_score = response.content.risky_behavior_score
                coach_recommendation = {
                    'Stocks': int(response.content.rec_stocks),
                    'Bonds': int(response.content.rec_bonds),
                    'Cash': int(response.content.rec_cash),
                    'Alternatives': int(response.content.rec_alternatives),
                }
        except Exception as e:
            feedback = f"AI coach error: {e}"
    if AGNO_READY and narrative_agent:
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
    st.session_state.coach_recommendation = coach_recommendation
    st.rerun() 