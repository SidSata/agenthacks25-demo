"""
Financial Portfolio Analyzer – v1
Streamlit app for learning asset allocation and compound interest
Run with:
    pip install streamlit pandas numpy agno openai
    streamlit run portfolio_sim_app.py

Environment:
  • Set your OpenAI key in .env or Streamlit Secrets →  OPENAI_API_KEY="sk‑..."

Learning Goal:
  • Learn how to allocate assets (stocks, bonds, cash) across your lifetime
  • See the power of compound interest and risk/reward tradeoffs
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field

# --- AGNO AGENT SETUP -------------------------------------------------------
class AllocationOutput(BaseModel):
    stocks: float = Field(..., description="% allocation to stocks (0-100)")
    bonds: float = Field(..., description="% allocation to bonds (0-100)")
    cash: float = Field(..., description="% allocation to cash (0-100)")
    explanation: str = Field(..., description="Rationale for this allocation")

OPENAI_API_KEY_VALUE = None
AGNO_READY = False
allocation_agent: Agent | None = None

def _initialize_agno_agent():
    global OPENAI_API_KEY_VALUE, AGNO_READY, allocation_agent
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
            allocation_agent = Agent(
                name="PortfolioAllocator",
                model=OpenAIChat(id="gpt-4o"),
                instructions=[
                    "You are a financial advisor specializing in lifetime asset allocation.",
                    "Given a user's age, risk tolerance, and investment horizon, suggest an allocation (percentages) to stocks, bonds, and cash.",
                    "Explain your rationale in simple terms for a beginner.",
                    "Output JSON with keys: stocks, bonds, cash, explanation. Percentages must sum to 100.",
                ],
                response_model=AllocationOutput,
            )
            AGNO_READY = True
        except Exception as e:
            st.warning(f"Failed to initialize Agno agent: {e}. Using fallback.")
            AGNO_READY = False
    else:
        st.warning("OpenAI key not provided. AI features will be disabled.")
        AGNO_READY = False

# --- SIMULATION LOGIC -------------------------------------------------------
def simulate_portfolio(initial, allocation, years, stock_return, bond_return, cash_return, stock_vol, bond_vol, cash_vol, annual_contrib=0):
    """
    Simulate portfolio growth over time with compound interest and annual rebalancing.
    Returns a DataFrame with yearly values for each asset and total.
    """
    np.random.seed(42)
    results = []
    stocks = initial * allocation['stocks'] / 100
    bonds = initial * allocation['bonds'] / 100
    cash = initial * allocation['cash'] / 100
    for year in range(years+1):
        total = stocks + bonds + cash
        results.append({
            'Year': year,
            'Stocks': stocks,
            'Bonds': bonds,
            'Cash': cash,
            'Total': total
        })
        # Simulate returns (lognormal for realism)
        stocks *= np.random.lognormal(stock_return, stock_vol)
        bonds  *= np.random.lognormal(bond_return, bond_vol)
        cash   *= np.random.lognormal(cash_return, cash_vol)
        # Add annual contribution
        stocks += annual_contrib * allocation['stocks'] / 100
        bonds  += annual_contrib * allocation['bonds'] / 100
        cash   += annual_contrib * allocation['cash'] / 100
        # Rebalance
        total = stocks + bonds + cash
        stocks = total * allocation['stocks'] / 100
        bonds  = total * allocation['bonds'] / 100
        cash   = total * allocation['cash'] / 100
    return pd.DataFrame(results)

# --- STREAMLIT UI -----------------------------------------------------------
st.set_page_config("💰 Portfolio Analyzer", layout="wide")
_initialize_agno_agent()

st.title("💰 Financial Portfolio Analyzer")
st.markdown("""
Learn how to allocate your investments across stocks, bonds, and cash as you age. See how compound interest grows your wealth!
""")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Your Age", 18, 70, 30)
    horizon = st.slider("Years to Simulate", 5, 50, 30)
    risk = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"], index=1)
    initial = st.number_input("Initial Investment ($)", 1000, 1_000_000, 10000, step=1000)
    annual_contrib = st.number_input("Annual Contribution ($)", 0, 100_000, 2000, step=500)
with col2:
    st.markdown("""
    **Risk Tolerance Guide:**
    - Low: Prefer safety, less volatility
    - Moderate: Balanced approach
    - High: Willing to accept more ups and downs for higher growth
    """)
    st.info("Try different ages and risk levels to see how advice and growth change!")

if st.button("Analyze Portfolio"):
    with st.spinner("Allocating assets and simulating growth..."):
        # AGNO agent for allocation
        allocation = {'stocks': 60, 'bonds': 30, 'cash': 10, 'explanation': 'Default 60/30/10 allocation.'}
        if AGNO_READY and allocation_agent:
            user_prompt = f"Age: {age}\nRisk tolerance: {risk}\nInvestment horizon: {horizon} years"
            try:
                response: RunResponse = allocation_agent.run(user_prompt)
                if response and response.content and isinstance(response.content, AllocationOutput):
                    allocation = {
                        'stocks': response.content.stocks,
                        'bonds': response.content.bonds,
                        'cash': response.content.cash,
                        'explanation': response.content.explanation
                    }
            except Exception as e:
                st.warning(f"AI agent error – using default allocation. {e}")
        st.subheader("Recommended Allocation")
        st.write(f"**Stocks:** {allocation['stocks']:.0f}%  |  **Bonds:** {allocation['bonds']:.0f}%  |  **Cash:** {allocation['cash']:.0f}%")
        st.caption(allocation['explanation'])
        # Simulate returns (historical averages, can be tweaked)
        if risk == "Low":
            stock_return, bond_return, cash_return = 0.04, 0.025, 0.01
            stock_vol, bond_vol, cash_vol = 0.10, 0.05, 0.01
        elif risk == "High":
            stock_return, bond_return, cash_return = 0.08, 0.03, 0.01
            stock_vol, bond_vol, cash_vol = 0.18, 0.08, 0.01
        else:
            stock_return, bond_return, cash_return = 0.06, 0.028, 0.01
            stock_vol, bond_vol, cash_vol = 0.14, 0.06, 0.01
        df = simulate_portfolio(
            initial, allocation, horizon,
            stock_return, bond_return, cash_return,
            stock_vol, bond_vol, cash_vol,
            annual_contrib=annual_contrib
        )
        st.subheader("Portfolio Growth Over Time")
        st.line_chart(df.set_index('Year')[['Stocks','Bonds','Cash','Total']], use_container_width=True, height=350)
        st.metric("Final Portfolio Value", f"${df['Total'].iloc[-1]:,.0f}")
        st.metric("Total Growth (%)", f"{(df['Total'].iloc[-1]/initial-1)*100:.1f}%")
        st.markdown("---")
        st.write("### Year-by-Year Breakdown")
        st.dataframe(df, use_container_width=True, height=350)
        st.markdown("---")
        st.write("#### What is Compound Interest?")
        st.info("Compound interest means you earn returns not just on your original investment, but also on the returns you've already earned. Over time, this can dramatically grow your wealth!") 