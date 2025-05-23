"""
Stock Market Showdown: Beat the Legends CLI Game – v1
A CLI game where you compete against Warren Buffett and George Soros in stock picking

Run with:
    pip install pandas numpy agno openai
    python stock_market_game.py

Environment:
  • Set your OpenAI key in .env or as environment variable →  OPENAI_API_KEY="sk‑..."

Game Premise:
  • You start in the year 2000 with $10,000
  • Compete against AI agents mimicking Warren Buffett and George Soros
  • Pick from 20 random stocks over 10 years (2000-2010)
  • Make the most money by deciding to buy, sell, or hold
"""

import os
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Agno imports
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field

# Terminal colors for better UI
try:
    from colorama import init, Fore, Back, Style
    init()  # Initialize colorama
    COLOR_ENABLED = True
except ImportError:
    # Fallback if colorama is not installed
    class DummyFore:
        def __getattr__(self, name):
            return ""
    class DummyBack:
        def __getattr__(self, name):
            return ""
    class DummyStyle:
        def __getattr__(self, name):
            return ""
    Fore = DummyFore()
    Back = DummyBack()
    Style = DummyStyle()
    COLOR_ENABLED = False

# Load environment variables
load_dotenv()

# --- GAME CONSTANTS -----------------------------------------------------------
START_YEAR = 2000
END_YEAR = 2010
YEARS = END_YEAR - START_YEAR
INITIAL_CAPITAL = 10000
NUM_STOCKS = 20

# --- USER CONTEXT ------------------------------------------------------------
# Player's risk profile and preferences
USER_CONTEXT = {
    "risk_profile": "risk_seeking",  # Options: risk_seeking, balanced, risk_averse, extremely_risk_averse
    "investment_goal": "growth",  # Options: growth, income, capital_preservation
    "investment_horizon": "medium_term",  # Options: short_term, medium_term, long_term
    "preferences": {
        "seeks": ["growth_stocks", "momentum_plays", "emerging_trends", "disruptive_technologies"],
        "avoids": ["slow_growth", "mature_industries", "overvalued_stocks"]
    }
}

# --- PYDANTIC MODELS ---------------------------------------------------------
class StockDecision(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    action: str = Field(..., description="Action to take: 'buy', 'sell', or 'hold'")
    quantity: int = Field(..., description="Number of shares to buy or sell")
    reasoning: str = Field(..., description="Reasoning behind the decision")

class StockPickerOutput(BaseModel):
    stocks: list[str] = Field(..., description="List of 20 stock ticker symbols")

class MarketEventOutput(BaseModel):
    headline: str = Field(..., description="Market headline for the year")
    description: str = Field(..., description="Brief description of the market events")
    impact: dict = Field(..., description="Impact on different sectors (tech, finance, energy, etc.)")

# --- AGNO AGENT SETUP --------------------------------------------------------
OPENAI_API_KEY_VALUE = None
AGNO_READY = False
buffett_agent = None
soros_agent = None
stock_picker_agent = None
market_event_agent = None

def initialize_agno_agents():
    """Initialize Agno agents with proper instructions."""
    global OPENAI_API_KEY_VALUE, AGNO_READY
    global buffett_agent, soros_agent, stock_picker_agent, market_event_agent
    
    try:
        import agno
    except ImportError:
        print("ERROR: 'agno' package not installed. Install with: pip install agno")
        AGNO_READY = False
        return
    
    OPENAI_API_KEY_VALUE = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY_VALUE:
        print("ERROR: OpenAI API key not found. Set it in .env file or as environment variable.")
        AGNO_READY = False
        return
    
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY_VALUE
    
    try:
        # Stock Picker Agent - selects 20 random stocks that existed in 2000
        stock_picker_agent = Agent(
            name="StockPickerAgent",
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "You are a stock market expert who selects a diverse set of 20 stocks that existed in the year 2000.",
                "Choose a mix of blue-chip companies, growth stocks, and value investments from different sectors.",
                "Only include actual companies that were publicly traded in 2000 with their correct ticker symbols.",
                "Do not include stocks that were not public in 2000 or didn't exist yet.",
                "Output a JSON list of exactly 20 stock ticker symbols."
            ],
            response_model=StockPickerOutput,
        )
        
        # Warren Buffett Agent - makes decisions based on Buffett's investment philosophy
        buffett_agent = Agent(
            name="WarrenBuffettAgent",
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "You are Warren Buffett in the year 2000. You have no knowledge of events after 2000.",
                "Make investment decisions based on your value investing philosophy:",
                "- Focus on companies with strong economic moats and competitive advantages",
                "- Look for businesses with consistent earning power and good return on equity",
                "- Prefer companies with honest and competent management",
                "- Insist on a margin of safety in purchase price",
                "- Hold for the long term when you find a great business",
                "- Avoid businesses you don't understand",
                "- See market downturns as buying opportunities",
                "- Concentrate investments in your best ideas rather than diversifying widely",
                "- Be fearful when others are greedy, and greedy when others are fearful",
                "Given the current year's market events and stock information, decide whether to buy, sell, or hold each stock in your portfolio.",
                "For each decision, provide your trademark folksy reasoning in your authentic voice.",
                "Use your well-known quotes and metaphors where appropriate.",
                "CRITICAL: Do not use knowledge of events or stock performance after the year 2000."
            ],
        )
        
        # George Soros Agent - makes decisions based on Soros's investment philosophy
        soros_agent = Agent(
            name="GeorgeSorosAgent",
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "You are George Soros in the year 2000. You have no knowledge of events after 2000.",
                "Make investment decisions based on your reflexivity theory and macro trading philosophy:",
                "- Focus on identifying large market trends and economic imbalances",
                "- Look for market bubbles and inefficiencies to exploit",
                "- Willing to make bold, high-conviction trades",
                "- Emphasize reflexivity - how market perceptions can create self-reinforcing trends",
                "- Take strong positions when you see significant market mispricing",
                "- Not afraid to short stocks or industries you believe are overvalued",
                "- Quick to reverse your position when your thesis appears wrong",
                "- Often front-run or anticipate central bank and government policy changes",
                "- Act aggressively when you see a 'fat tail' opportunity (rare, high-impact events)",
                "Given the current year's market events and stock information, decide whether to buy, sell, or hold each stock in your portfolio.",
                "For each decision, provide your reasoning in your authentic voice with philosophical undertones.",
                "Include references to market psychology, perception/reality feedback loops, and macro trends where relevant.",
                "Use your trademark contrarian approach when appropriate.",
                "CRITICAL: Do not use knowledge of events or stock performance after the year 2000."
            ],
        )
        
        # Market Event Agent - creates realistic market events for each year
        market_event_agent = Agent(
            name="MarketEventAgent",
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "You are a financial historian creating accurate market events for each year from 2000 to 2010.",
                "For each year, provide a realistic headline, description, and impact on different market sectors.",
                "Use actual historical events from that specific year only.",
                "The description should be 1-2 sentences about major market movements and economic factors.",
                "Impact should include effects on key sectors: tech, finance, energy, healthcare, consumer, industrial, etc.",
                "Output JSON with keys: headline, description, and impact (a dictionary of sectors and impact values from -10 to +10)."
            ],
            response_model=MarketEventOutput,
        )
        
        AGNO_READY = True
    except Exception as e:
        print(f"Failed to initialize Agno agents: {e}")
        AGNO_READY = False

# --- STOCK DATA SIMULATION ---------------------------------------------------
def generate_stock_data(tickers, start_year=START_YEAR, end_year=END_YEAR):
    """
    Generate realistic stock data for the given tickers from start_year to end_year.
    Returns a DataFrame with yearly prices for each ticker.
    """
    years = list(range(start_year, end_year + 1))
    data = {'Year': years}
    
    # For each ticker, generate a price series with realistic growth and volatility
    for ticker in tickers:
        # Randomize starting price between $10 and $100
        start_price = random.uniform(10, 100)
        
        # Different stocks have different growth and volatility characteristics
        annual_return = random.normalvariate(0.08, 0.04)  # Mean 8%, std 4%
        volatility = random.uniform(0.15, 0.40)  # 15% to 40% annual volatility
        
        # Generate price series
        prices = [start_price]
        for i in range(1, len(years)):
            # Add some randomness to annual returns
            year_return = random.normalvariate(annual_return, volatility)
            
            # Some stocks might have big swings (boom or bust)
            if random.random() < 0.1:  # 10% chance of a big event
                year_return = year_return * random.uniform(-3, 3)
                
            # Adjust for dot-com crash (2000-2002) and financial crisis (2008)
            if years[i] in [2000, 2001, 2002] and 'tech' in ticker.lower():
                year_return -= random.uniform(0.1, 0.5)
            if years[i] == 2008:
                year_return -= random.uniform(0.2, 0.4)
                
            # Calculate new price
            new_price = prices[-1] * (1 + year_return)
            prices.append(max(0.1, new_price))  # Ensure price doesn't go below $0.1
            
        data[ticker] = prices
    
    return pd.DataFrame(data)

# --- GAME LOGIC --------------------------------------------------------------
class Player:
    def __init__(self, name, capital=INITIAL_CAPITAL):
        self.name = name
        self.cash = capital
        self.portfolio = {}  # {ticker: quantity}
        self.history = []  # List of transactions
        self.net_worth_history = [capital]  # Track net worth by year
    
    def buy(self, ticker, quantity, price, year, reasoning=""):
        cost = quantity * price
        if cost > self.cash:
            quantity = int(self.cash / price)  # Buy as many as possible
            cost = quantity * price
            
        if quantity <= 0:
            return False
            
        self.cash -= cost
        if ticker in self.portfolio:
            self.portfolio[ticker] += quantity
        else:
            self.portfolio[ticker] = quantity
            
        self.history.append({
            'year': year,
            'ticker': ticker,
            'action': 'buy',
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'reasoning': reasoning
        })
        return True
    
    def sell(self, ticker, quantity, price, year, reasoning=""):
        if ticker not in self.portfolio or self.portfolio[ticker] < quantity:
            quantity = self.portfolio.get(ticker, 0)  # Sell as many as possible
            
        if quantity <= 0:
            return False
            
        revenue = quantity * price
        self.cash += revenue
        self.portfolio[ticker] -= quantity
        
        if self.portfolio[ticker] == 0:
            del self.portfolio[ticker]
            
        self.history.append({
            'year': year,
            'ticker': ticker,
            'action': 'sell',
            'quantity': quantity,
            'price': price,
            'revenue': revenue,
            'reasoning': reasoning
        })
        return True
    
    def hold(self, ticker, year, reasoning=""):
        if ticker in self.portfolio and self.portfolio[ticker] > 0:
            self.history.append({
                'year': year,
                'ticker': ticker,
                'action': 'hold',
                'quantity': self.portfolio.get(ticker, 0),
                'reasoning': reasoning
            })
            return True
        return False
    
    def calculate_net_worth(self, stock_prices):
        """Calculate current net worth based on cash and portfolio value."""
        portfolio_value = sum(self.portfolio.get(ticker, 0) * price 
                             for ticker, price in stock_prices.items())
        return self.cash + portfolio_value

    def update_net_worth_history(self, stock_prices):
        """Update net worth history for the current year."""
        net_worth = self.calculate_net_worth(stock_prices)
        self.net_worth_history.append(net_worth)
        return net_worth

# --- AGENTS DECISIONS --------------------------------------------------------
def get_buffett_decisions(portfolio, stock_data, current_year, market_event):
    """Get Warren Buffett's investment decisions for the current year."""
    if not AGNO_READY or not buffett_agent:
        # Fallback to deterministic decisions if agent not available
        return buffett_fallback_decisions(portfolio, stock_data, current_year)
    
    decisions = []
    current_prices = {ticker: stock_data.loc[stock_data['Year'] == current_year, ticker].values[0] 
                     for ticker in portfolio.keys()}
    
    # Prepare the portfolio information for the agent
    portfolio_info = []
    for ticker, quantity in portfolio.items():
        if quantity > 0:
            price = current_prices[ticker]
            value = price * quantity
            portfolio_info.append(f"{ticker}: {quantity} shares at ${price:.2f}/share = ${value:.2f}")
    
    # Prepare stock information
    stock_info = []
    for ticker in portfolio.keys():
        price = current_prices[ticker]
        price_last_year = stock_data.loc[stock_data['Year'] == current_year-1, ticker].values[0] if current_year > START_YEAR else price
        pct_change = ((price - price_last_year) / price_last_year * 100) if price_last_year > 0 else 0
        stock_info.append(f"{ticker}: ${price:.2f} ({pct_change:+.1f}%)")
    
    # Prepare prompt for Buffett agent
    prompt = f"""
Year: {current_year}
Cash: ${portfolio['cash']:.2f}

Market Event:
{market_event['headline']}
{market_event['description']}

Your Current Portfolio:
{'\n'.join(portfolio_info)}

Current Stock Prices:
{'\n'.join(stock_info)}

As Warren Buffett in {current_year}, what are your investment decisions?
For each stock in your portfolio, decide whether to buy more, sell some, or hold.
Include your reasoning for each decision in your folksy style.
"""
    
    try:
        response = buffett_agent.run(prompt)
        if response and response.content:
            # Parse Buffett's response to extract decisions
            raw_decisions = response.content
            
            # Simple parsing strategy: look for lines with BUY, SELL, or HOLD
            for line in raw_decisions.split('\n'):
                for ticker in portfolio.keys():
                    if ticker in line:
                        if "BUY" in line.upper():
                            # Extract quantity and reasoning
                            quantity = 10  # Default quantity
                            action = "buy"
                            reasoning = line
                        elif "SELL" in line.upper():
                            quantity = portfolio.get(ticker, 0) // 2  # Sell half by default
                            action = "sell"
                            reasoning = line
                        else:
                            quantity = 0
                            action = "hold"
                            reasoning = line
                        
                        decisions.append({
                            "ticker": ticker,
                            "action": action,
                            "quantity": quantity,
                            "reasoning": reasoning
                        })
                        break
            
            # If no decisions were extracted, use fallback
            if not decisions:
                return buffett_fallback_decisions(portfolio, stock_data, current_year)
                
            return decisions
        else:
            return buffett_fallback_decisions(portfolio, stock_data, current_year)
    except Exception as e:
        print(f"Error getting Buffett's decisions: {e}")
        return buffett_fallback_decisions(portfolio, stock_data, current_year)

def buffett_fallback_decisions(portfolio, stock_data, current_year):
    """Fallback logic for Warren Buffett's investment decisions."""
    decisions = []
    cash = portfolio['cash']
    
    for ticker, quantity in portfolio.items():
        if ticker == 'cash':
            continue
            
        # Get current price
        current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
        
        # Get previous year price if available
        if current_year > START_YEAR:
            prev_price = stock_data.loc[stock_data['Year'] == current_year-1, ticker].values[0]
            price_change = (current_price - prev_price) / prev_price
        else:
            price_change = 0
            
        # Buffett tends to hold quality companies for the long term
        action = "hold"
        buy_quantity = 0
        sell_quantity = 0
        
        # If we don't own the stock, consider buying it
        if quantity == 0:
            if random.random() < 0.3 and cash > current_price * 10:
                action = "buy"
                buy_quantity = min(int(cash / (4 * current_price)), 20)  # Use 1/4 of cash
                reasoning = f"I'm establishing a new position in {ticker}. This appears to be a quality business at a reasonable price."
            else:
                action = "hold"
                reasoning = f"I'm monitoring {ticker} but don't see a compelling entry point at current levels."
        else:
            # We already own this stock - apply normal Buffett logic
            # More thoughtful reasoning based on ticker and market conditions
            buffett_phrases = [
                f"I believe {ticker} has the characteristics of a business with a durable competitive advantage.",
                f"With {ticker}, I'm looking for consistent earning power and good returns on equity.",
                f"The management of {ticker} appears competent and shareholder-oriented.",
                f"I prefer to hold wonderful companies like {ticker} at fair prices for the long term.",
                f"When evaluating {ticker}, I focus on the company's ability to maintain its economic moat."
            ]
            
            # Buffett is more likely to buy on market crashes (value opportunity)
            market_event = market_event_fallback(current_year)
            market_impact = sum(market_event['impact'].values()) / len(market_event['impact'])
            
            # Buffett buys more aggressively during market downturns if he has cash
            if market_impact < -3 and cash > current_price * 10:
                buy_probability = 0.6
            elif price_change < -0.15 and cash > current_price * 10:  # Stock specific downturn
                buy_probability = 0.5
            else:
                buy_probability = 0.2
                
            # Decide action based on probabilities
            if random.random() < buy_probability and cash > current_price * 10:
                action = "buy"
                buy_quantity = min(int(cash / (2 * current_price)), 10)  # Use up to half of cash
                
                # Buffett-style reasoning for buying
                buy_reasons = [
                    f"Be fearful when others are greedy, and greedy when others are fearful. I see fear in {ticker}, which creates opportunity.",
                    f"With {ticker}, I see a wonderful company available at a fair price after recent market activity.",
                    f"The recent decline in {ticker} appears to be short-term thinking by the market, not a fundamental change in the business.",
                    f"Our favorite holding period is forever. I'm adding to our position in {ticker} with the intention of holding for many years.",
                    f"Price is what you pay, value is what you get. I believe {ticker} offers good value at this price."
                ]
                reasoning = random.choice(buy_reasons)
            
            # Occasionally sell if in a specific sector experiencing troubles
            # Buffett tends to avoid tech stocks in this period
            elif (('tech' in ticker.lower() and current_year <= 2002) or  # Tech crash
                  ('finance' in ticker.lower() and current_year >= 2007) or  # Financial crisis
                  random.random() < 0.08):  # Small chance to sell anything
                action = "sell"
                sell_quantity = quantity // 2  # Sell half
                
                # Buffett-style reasoning for selling
                sell_reasons = [
                    f"When the fundamentals of a business deteriorate, it's time to reevaluate our position in {ticker}.",
                    f"I've become concerned about changes in the competitive landscape surrounding {ticker}.",
                    f"Rule #1: Never lose money. Rule #2: Never forget rule #1. I'm reducing our exposure to {ticker} to protect capital.",
                    f"It's only when the tide goes out that you learn who's been swimming naked. I see some concerning signs with {ticker}.",
                    f"I prefer to exit {ticker} now, as I see better opportunities to allocate our capital elsewhere."
                ]
                reasoning = random.choice(sell_reasons)
            else:
                # Hold reasoning
                hold_reasons = [
                    f"Benign neglect, bordering on sloth, remains the hallmark of our investment process. I'm comfortable continuing to hold {ticker}.",
                    f"The stock market is designed to transfer money from the active to the patient. I remain patient with our {ticker} position.",
                    f"Time is the friend of the wonderful business, the enemy of the mediocre. {ticker} remains a wonderful business.",
                    f"I see no reason to make changes to our {ticker} position at this time.",
                    f"Our favorite holding period is forever, and {ticker} continues to meet our criteria for a long-term holding."
                ]
                reasoning = random.choice(hold_reasons)
        
        decisions.append({
            "ticker": ticker,
            "action": action,
            "quantity": buy_quantity if action == "buy" else sell_quantity,
            "reasoning": reasoning
        })
    
    return decisions

def get_soros_decisions(portfolio, stock_data, current_year, market_event):
    """Get George Soros's investment decisions for the current year."""
    if not AGNO_READY or not soros_agent:
        # Fallback to deterministic decisions if agent not available
        return soros_fallback_decisions(portfolio, stock_data, current_year)
    
    decisions = []
    current_prices = {ticker: stock_data.loc[stock_data['Year'] == current_year, ticker].values[0] 
                     for ticker in portfolio.keys()}
    
    # Prepare the portfolio information for the agent
    portfolio_info = []
    for ticker, quantity in portfolio.items():
        if quantity > 0:
            price = current_prices[ticker]
            value = price * quantity
            portfolio_info.append(f"{ticker}: {quantity} shares at ${price:.2f}/share = ${value:.2f}")
    
    # Prepare stock information
    stock_info = []
    for ticker in portfolio.keys():
        price = current_prices[ticker]
        price_last_year = stock_data.loc[stock_data['Year'] == current_year-1, ticker].values[0] if current_year > START_YEAR else price
        pct_change = ((price - price_last_year) / price_last_year * 100) if price_last_year > 0 else 0
        stock_info.append(f"{ticker}: ${price:.2f} ({pct_change:+.1f}%)")
    
    # Prepare prompt for Soros agent
    prompt = f"""
Year: {current_year}
Cash: ${portfolio['cash']:.2f}

Market Event:
{market_event['headline']}
{market_event['description']}

Your Current Portfolio:
{'\n'.join(portfolio_info)}

Current Stock Prices:
{'\n'.join(stock_info)}

As George Soros in {current_year}, what are your investment decisions?
For each stock in your portfolio, decide whether to buy more, sell some, or hold.
Consider market reflexivity, macro trends, and potential inflection points.
Include your reasoning for each decision with philosophical undertones where relevant.
"""
    
    try:
        response = soros_agent.run(prompt)
        if response and response.content:
            # Parse Soros's response to extract decisions
            raw_decisions = response.content
            
            # Simple parsing strategy: look for lines with BUY, SELL, or HOLD
            for line in raw_decisions.split('\n'):
                for ticker in portfolio.keys():
                    if ticker in line:
                        if "BUY" in line.upper():
                            # Extract quantity and reasoning
                            quantity = 20  # Soros makes decisive moves
                            action = "buy"
                            reasoning = line
                        elif "SELL" in line.upper():
                            quantity = portfolio.get(ticker, 0) // 2  # Sell half by default
                            action = "sell"
                            reasoning = line
                        else:
                            quantity = 0
                            action = "hold"
                            reasoning = line
                        
                        decisions.append({
                            "ticker": ticker,
                            "action": action,
                            "quantity": quantity,
                            "reasoning": reasoning
                        })
                        break
            
            # If no decisions were extracted, use fallback
            if not decisions:
                return soros_fallback_decisions(portfolio, stock_data, current_year)
                
            return decisions
        else:
            return soros_fallback_decisions(portfolio, stock_data, current_year)
    except Exception as e:
        print(f"Error getting Soros's decisions: {e}")
        return soros_fallback_decisions(portfolio, stock_data, current_year)

def soros_fallback_decisions(portfolio, stock_data, current_year):
    """Fallback logic for George Soros's investment decisions."""
    decisions = []
    cash = portfolio['cash']
    
    for ticker, quantity in portfolio.items():
        if ticker == 'cash':
            continue
            
        # Get current price
        current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
        
        # Get previous year price if available
        if current_year > START_YEAR:
            prev_price = stock_data.loc[stock_data['Year'] == current_year-1, ticker].values[0]
            price_change = (current_price - prev_price) / prev_price
        else:
            price_change = 0
            
        # Soros tends to be more macro-focused, looking for major trends and reflexivity
        action = "hold"
        buy_quantity = 0
        sell_quantity = 0
        
        # If we don't own the stock, consider buying it based on macro trends
        if quantity == 0:
            if random.random() < 0.4 and cash > current_price * 5:
                action = "buy"
                buy_quantity = min(int(cash * 0.3 / current_price), 25)  # Use 30% of cash
                reasoning = f"I see potential reflexive opportunities with {ticker}. Market dynamics suggest this could benefit from emerging trends."
            else:
                action = "hold"
                reasoning = f"I'm analyzing {ticker} for potential macro dislocations but see no immediate catalyst."
        else:
            # We already own this stock - apply normal Soros logic
            # Check sector orientation and market dynamics
            is_tech = any(keyword in ticker.lower() for keyword in 
                         ["tech", "net", "soft", "sys", "data", "micro", "com"])
            is_financial = any(keyword in ticker.lower() for keyword in 
                              ["bank", "fin", "jp", "gs", "ms", "citi", "bac", "wfc"])
            is_cyclical = any(keyword in ticker.lower() for keyword in 
                             ["auto", "steel", "retail", "energy", "oil", "gas"])
            
            # Soros is a macro trader who looks for big shifts and inefficiencies
            # Get market event information for the year to inform decisions
            market_event = market_event_fallback(current_year)
            market_impact = sum(market_event['impact'].values()) / len(market_event['impact'])
            
            # Initialize reasoning with a default
            reasoning = "The market's perception is creating a reflexive reality that I'm monitoring closely."
            
            # Major trend detection - Soros often goes against consensus
            # but rides momentum once a trend is established
            if market_impact < -2:  # Significant market distress
                # Soros often finds opportunities in crisis
                buy_probability = 0.3  # Not as likely as value investors to buy broad dips
                sell_probability = 0.2  # Might be short already
                
                # Sometimes contrarian, sometimes momentum-based
                if is_financial and current_year in [2007, 2008]:  # Financial crisis years
                    sell_probability = 0.8  # Very likely to sell or short financials
                elif is_tech and current_year in [2000, 2001]:  # Dot-com burst years
                    sell_probability = 0.7  # Very likely to sell tech
                    
                # Soros's reasoning during market distress
                buy_reasons = [
                    f"I see a potential reflexive scenario developing with {ticker} where market pessimism has gone too far, creating asymmetric risk/reward.",
                    f"The current market dislocation in {ticker} represents a mispricing I can exploit. The crowd is acting irrationally.",
                    f"I believe {ticker} has been caught in a negative feedback loop that will soon reverse, creating a profitable inflection point.",
                    f"While the market has been punishing {ticker}, my analysis suggests the fundamentals remain stronger than perceived.",
                    f"The theory of reflexivity suggests that the negative perception of {ticker} has created an unwarranted reality, offering an opportunity."
                ]
                
                sell_reasons = [
                    f"I'm establishing a short position in {ticker} as I believe the negative reflexive process has further to run.",
                    f"My analysis of {ticker} suggests we're seeing the beginning of a major trend reversal that will accelerate to the downside.",
                    f"I believe {ticker} is caught in a negative feedback loop that will continue to pressure valuations downward.",
                    f"The market perception of {ticker} is still too optimistic relative to the changing economic reality.",
                    f"I'm positioning against {ticker} as I anticipate the full recognition of structural problems still ahead."
                ]
                
                hold_reasons = [
                    f"While I'm generally cautious in this market, {ticker} has unique qualities that warrant maintaining our position.",
                    f"I'm watching {ticker} closely for signs of capitulation which would offer a better entry point.",
                    f"The reflexive relationship between perception and reality in {ticker} is at an equilibrium point.",
                    f"I see balanced forces acting on {ticker}, making neither a buy nor sell compelling at this juncture.",
                    f"I'm maintaining a neutral stance on {ticker} while I assess the evolving macro environment."
                ]
                
                if price_change < -0.15 and random.random() < buy_probability and cash > current_price * 10:
                    # Soros sometimes makes aggressive contrarian bets during extreme pessimism
                    action = "buy"
                    buy_quantity = min(int(cash * 0.4 / current_price), 30)  # More concentrated positions
                    reasoning = random.choice(buy_reasons)
                elif random.random() < sell_probability:
                    action = "sell"
                    sell_quantity = max(1, quantity)  # Potentially exit entirely
                    reasoning = random.choice(sell_reasons)
                else:
                    reasoning = random.choice(hold_reasons)
            
            elif market_impact > 2:  # Strong positive market 
                # Soros might ride momentum or look for bubbles to short
                bubble_probability = 0.3
                
                if is_tech and current_year == 2000:  # Dot-com peak
                    bubble_probability = 0.7  # High chance to spot a bubble
                    
                if random.random() < bubble_probability:
                    action = "sell"
                    sell_quantity = max(1, quantity // 2)
                    reasoning = f"I believe {ticker} is exhibiting bubble-like behavior. The reflexive process that drove it higher is exhausting itself and will soon reverse."
                elif random.random() < 0.4 and cash > current_price * 5:
                    # Sometimes rides momentum
                    action = "buy"
                    buy_quantity = min(int(cash * 0.3 / current_price), 15)
                    reasoning = f"While the trend in {ticker} may be approaching extremes, the momentum could continue longer than most expect. I'm participating but will remain vigilant for signs of reversal."
                else:
                    reasoning = f"I'm watching {ticker} closely for signs of trend exhaustion. In reflexivity terms, we're in a phase where perception and reality are reinforcing each other."
            
            else:  # Neutral market
                # Look for specific sector opportunities
                if price_change > 0.2:  # Strong momentum
                    # Might ride or fade momentum depending on conviction
                    if random.random() < 0.4:
                        action = "sell"
                        sell_quantity = max(1, quantity // 3)
                        reasoning = f"The rapid appreciation in {ticker} suggests a reflexive process that may be approaching its limits. I'm taking some profits while monitoring for further developments."
                    else:
                        reasoning = f"While {ticker} has moved significantly, my analysis suggests this trend still has room to run based on market psychology and fundamental factors."
                elif price_change < -0.2:  # Significant drop
                    if random.random() < 0.5 and cash > current_price * 8:
                        action = "buy"
                        buy_quantity = min(int(cash * 0.25 / current_price), 20)
                        reasoning = f"The sell-off in {ticker} appears overdone. The market's perception has created a reality that doesn't match my fundamental analysis, creating an opportunity."
                    else:
                        reasoning = f"I'm assessing whether the decline in {ticker} is the beginning of a new trend or a temporary deviation. The theory of reflexivity suggests monitoring the feedback loop between perception and fundamentals."
                else:
                    # Philosophical hold reasoning
                    hold_reasons = [
                        f"The current equilibrium in {ticker} does not present a compelling case for action based on my analysis of market reflexivity.",
                        f"I'm maintaining our position in {ticker} while watching for catalysts that could trigger a new reflexive process.",
                        f"My assessment of {ticker} suggests we're in a period where market perceptions and reality are relatively balanced.",
                        f"I see no significant mispricing in {ticker} at this juncture that would warrant either accumulation or distribution.",
                        f"I'm continuing to hold {ticker} while watching for potential policy changes or market shifts that could alter its trajectory."
                    ]
                    reasoning = random.choice(hold_reasons)
        
        decisions.append({
            "ticker": ticker,
            "action": action,
            "quantity": buy_quantity if action == "buy" else sell_quantity,
            "reasoning": reasoning
        })
    
    return decisions

def get_market_event(year):
    """Get market event for the specified year."""
    if not AGNO_READY or not market_event_agent:
        return market_event_fallback(year)
    
    prompt = f"Create a historically accurate market event for the year {year}. Include headline, brief description, and sector impacts."
    
    try:
        response: RunResponse = market_event_agent.run(prompt)
        if response and response.content and isinstance(response.content, MarketEventOutput):
            return {
                "headline": response.content.headline,
                "description": response.content.description,
                "impact": response.content.impact
            }
        else:
            return market_event_fallback(year)
    except Exception as e:
        print(f"Error getting market event: {e}")
        return market_event_fallback(year)

def market_event_fallback(year):
    """Fallback market events based on historical data."""
    events = {
        2000: {
            "headline": "Dot-Com Bubble Begins to Burst",
            "description": "Tech stocks start a dramatic decline as investors question the valuation of internet companies.",
            "impact": {"tech": -8, "finance": -3, "energy": 2, "healthcare": 0, "consumer": -2, "industrial": -1}
        },
        2001: {
            "headline": "9/11 Attacks Shock Markets",
            "description": "Terrorist attacks cause market shutdown and significant drops when reopened.",
            "impact": {"tech": -5, "finance": -7, "energy": -3, "healthcare": -2, "consumer": -6, "industrial": -4, "travel": -9}
        },
        2002: {
            "headline": "Corporate Accounting Scandals Rock Markets",
            "description": "Enron and WorldCom bankruptcies shatter investor confidence.",
            "impact": {"tech": -6, "finance": -8, "energy": -5, "healthcare": -2, "consumer": -3, "industrial": -4}
        },
        2003: {
            "headline": "Markets Begin Recovery",
            "description": "Stocks start to rebound as economy recovers from dot-com crash.",
            "impact": {"tech": 4, "finance": 3, "energy": 5, "healthcare": 4, "consumer": 3, "industrial": 5}
        },
        2004: {
            "headline": "Fed Begins Rate Hike Cycle",
            "description": "Federal Reserve starts raising interest rates for the first time in four years.",
            "impact": {"tech": 2, "finance": -1, "energy": 3, "healthcare": 1, "consumer": 0, "industrial": 1, "real-estate": -3}
        },
        2005: {
            "headline": "Housing Boom Continues",
            "description": "Real estate market reaches new heights as housing prices soar.",
            "impact": {"tech": 1, "finance": 4, "energy": 2, "healthcare": 0, "consumer": 3, "industrial": 2, "real-estate": 7}
        },
        2006: {
            "headline": "Housing Market Shows Signs of Weakness",
            "description": "Early indicators of housing market troubles begin to appear.",
            "impact": {"tech": 0, "finance": -2, "energy": 4, "healthcare": 1, "consumer": -1, "industrial": 0, "real-estate": -4}
        },
        2007: {
            "headline": "Subprime Mortgage Crisis Begins",
            "description": "Housing bubble bursts as subprime mortgage defaults rise dramatically.",
            "impact": {"tech": -2, "finance": -6, "energy": 3, "healthcare": 0, "consumer": -4, "industrial": -3, "real-estate": -8}
        },
        2008: {
            "headline": "Global Financial Crisis",
            "description": "Lehman Brothers collapse triggers worldwide financial panic and recession.",
            "impact": {"tech": -7, "finance": -9, "energy": -8, "healthcare": -4, "consumer": -7, "industrial": -6, "real-estate": -9}
        },
        2009: {
            "headline": "Markets Hit Bottom and Begin Recovery",
            "description": "Stocks reach their lowest point in March before starting a long bull run.",
            "impact": {"tech": 5, "finance": 3, "energy": 4, "healthcare": 4, "consumer": 2, "industrial": 3, "real-estate": 0}
        },
        2010: {
            "headline": "Recovery Continues Despite European Debt Crisis",
            "description": "Markets continue upward trend despite concerns about European sovereign debt.",
            "impact": {"tech": 4, "finance": 2, "energy": 3, "healthcare": 3, "consumer": 3, "industrial": 4, "real-estate": 1}
        }
    }
    
    return events.get(year, {
        "headline": f"Market Events for {year}",
        "description": "Steady market conditions with normal fluctuations.",
        "impact": {"tech": 0, "finance": 0, "energy": 0, "healthcare": 0}
    })

def select_random_stocks():
    """Select 20 random stocks that existed in 2000."""
    if not AGNO_READY or not stock_picker_agent:
        return fallback_stocks()
    
    prompt = "Select 20 diverse stocks that existed in the year 2000, including their correct ticker symbols."
    
    try:
        response: RunResponse = stock_picker_agent.run(prompt)
        if response and response.content and isinstance(response.content, StockPickerOutput):
            return response.content.stocks
        else:
            return fallback_stocks()
    except Exception as e:
        print(f"Error selecting random stocks: {e}")
        return fallback_stocks()

def fallback_stocks():
    """Fallback list of stocks that existed in 2000."""
    return [
        "MSFT",  # Microsoft
        "AAPL",  # Apple
        "INTC",  # Intel
        "CSCO",  # Cisco
        "IBM",   # IBM
        "GE",    # General Electric
        "XOM",   # Exxon Mobil
        "WMT",   # Walmart
        "PG",    # Procter & Gamble
        "JNJ",   # Johnson & Johnson
        "KO",    # Coca-Cola
        "MRK",   # Merck
        "PFE",   # Pfizer
        "JPM",   # JPMorgan Chase
        "C",     # Citigroup
        "BAC",   # Bank of America
        "DIS",   # Disney
        "HD",    # Home Depot
        "MCD",   # McDonald's
        "T"      # AT&T
    ]

# --- CLI INTERFACE -----------------------------------------------------------
def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print game header."""
    print("\n" + Fore.CYAN + "=" * 80 + Style.RESET_ALL)
    print(Fore.CYAN + " " * 25 + Style.BRIGHT + "STOCK MARKET SHOWDOWN" + Style.RESET_ALL)
    print(Fore.CYAN + " " * 20 + "Beat Warren Buffett & George Soros" + Style.RESET_ALL)
    print(Fore.CYAN + "=" * 80 + Style.RESET_ALL)

def print_year_header(year, market_event):
    """Print year header with market event."""
    print("\n" + Fore.YELLOW + "=" * 80 + Style.RESET_ALL)
    print(Fore.YELLOW + f" YEAR: {year} ".center(80, "*") + Style.RESET_ALL)
    print(Fore.RED + f" {market_event['headline']} ".center(80, "-") + Style.RESET_ALL)
    print(Fore.WHITE + f" {market_event['description']} ".center(80, " ") + Style.RESET_ALL)
    print(Fore.YELLOW + "=" * 80 + Style.RESET_ALL + "\n")

def print_portfolio(player, stock_data, current_year):
    """Print player's portfolio."""
    print(f"\n{Fore.GREEN + Style.BRIGHT}{player.name}'s Portfolio:".upper() + Style.RESET_ALL)
    print(Fore.GREEN + "-" * 60 + Style.RESET_ALL)
    
    total_value = player.cash
    print(f"{Fore.GREEN}Cash: ${player.cash:.2f}{Style.RESET_ALL}")
    
    if player.portfolio:
        print(f"\n{Fore.GREEN}Stock Holdings:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'Ticker':<8} {'Shares':<8} {'Price':<10} {'Value':<12} {'Change':<10}{Style.RESET_ALL}")
        print(Fore.GREEN + "-" * 60 + Style.RESET_ALL)
        
        for ticker, quantity in player.portfolio.items():
            current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
            value = quantity * current_price
            total_value += value
            
            # Calculate price change if not first year
            if current_year > START_YEAR:
                prev_price = stock_data.loc[stock_data['Year'] == current_year-1, ticker].values[0]
                change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                if change_pct > 0:
                    change_str = f"{Fore.GREEN}{change_pct:+.1f}%{Style.RESET_ALL}"
                elif change_pct < 0:
                    change_str = f"{Fore.RED}{change_pct:+.1f}%{Style.RESET_ALL}"
                else:
                    change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"
                
            print(f"{ticker:<8} {quantity:<8} ${current_price:<9.2f} ${value:<11.2f} {change_str:<10}")
    
    print(Fore.GREEN + "-" * 60 + Style.RESET_ALL)
    print(f"{Fore.GREEN + Style.BRIGHT}Total Net Worth: ${total_value:.2f}{Style.RESET_ALL}")
    print(Fore.GREEN + "-" * 60 + Style.RESET_ALL)

def print_leaderboard(player, buffett, soros, current_year, stock_data):
    """Print leaderboard showing performance comparison."""
    print("\n" + Fore.MAGENTA + "=" * 40 + Style.RESET_ALL)
    print(Fore.MAGENTA + Style.BRIGHT + " LEADERBOARD ".center(40, "*") + Style.RESET_ALL)
    print(Fore.MAGENTA + "=" * 40 + Style.RESET_ALL)
    
    # Calculate current net worth for all players
    player_worth = player.calculate_net_worth({ticker: stock_data.loc[stock_data['Year'] == current_year, ticker].values[0] 
                                              for ticker in stock_data.columns if ticker != 'Year'})
    buffett_worth = buffett.calculate_net_worth({ticker: stock_data.loc[stock_data['Year'] == current_year, ticker].values[0] 
                                               for ticker in stock_data.columns if ticker != 'Year'})
    soros_worth = soros.calculate_net_worth({ticker: stock_data.loc[stock_data['Year'] == current_year, ticker].values[0] 
                                         for ticker in stock_data.columns if ticker != 'Year'})
    
    # Calculate growth from initial investment
    player_growth = (player_worth / INITIAL_CAPITAL - 1) * 100
    buffett_growth = (buffett_worth / INITIAL_CAPITAL - 1) * 100
    soros_growth = (soros_worth / INITIAL_CAPITAL - 1) * 100
    
    # Sort players by net worth
    players = [
        (player.name, player_worth, player_growth),
        ("Warren Buffett", buffett_worth, buffett_growth),
        ("George Soros", soros_worth, soros_growth)
    ]
    players.sort(key=lambda x: x[1], reverse=True)
    
    # Print ranking
    for i, (name, worth, growth) in enumerate(players, 1):
        # Highlight player's name
        if name == player.name:
            name_str = f"{Fore.CYAN + Style.BRIGHT}{name:<15}{Style.RESET_ALL}"
        elif name == "Warren Buffett":
            name_str = f"{Fore.YELLOW}{name:<15}{Style.RESET_ALL}"
        else:  # George Soros
            name_str = f"{Fore.MAGENTA}{name:<15}{Style.RESET_ALL}"
            
        # Color growth based on positive/negative
        if growth > 0:
            growth_str = f"({Fore.GREEN}{growth:+.1f}%{Style.RESET_ALL})"
        elif growth < 0:
            growth_str = f"({Fore.RED}{growth:+.1f}%{Style.RESET_ALL})"
        else:
            growth_str = f"({growth:+.1f}%)"
            
        # Add medal for ranking
        if i == 1:
            rank = f"{Fore.YELLOW}🥇 {i}.{Style.RESET_ALL}"
        elif i == 2:
            rank = f"{Fore.WHITE}🥈 {i}.{Style.RESET_ALL}"
        else:
            rank = f"{Fore.RED}🥉 {i}.{Style.RESET_ALL}"
            
        print(f"{rank} {name_str} ${worth:<10.2f} {growth_str}")
    
    print(Fore.MAGENTA + "=" * 40 + Style.RESET_ALL)

def print_decision_history(player, year):
    """Print decision history for the specified year."""
    print(f"\n{Fore.BLUE + Style.BRIGHT}Decisions in {year}:{Style.RESET_ALL}")
    print(Fore.BLUE + "-" * 60 + Style.RESET_ALL)
    
    year_decisions = [transaction for transaction in player.history if transaction['year'] == year]
    
    if not year_decisions:
        print(f"{Fore.YELLOW}No transactions this year.{Style.RESET_ALL}")
        return
    
    for transaction in year_decisions:
        action = transaction['action'].upper()
        ticker = transaction['ticker']
        quantity = transaction['quantity']
        
        if action == 'BUY':
            price = transaction['price']
            cost = transaction['cost']
            print(f"{Fore.GREEN + Style.BRIGHT}{action}:{Style.RESET_ALL} {quantity} shares of {Fore.YELLOW}{ticker}{Style.RESET_ALL} at ${price:.2f} (Total: ${cost:.2f})")
        elif action == 'SELL':
            price = transaction['price']
            revenue = transaction['revenue']
            print(f"{Fore.RED + Style.BRIGHT}{action}:{Style.RESET_ALL} {quantity} shares of {Fore.YELLOW}{ticker}{Style.RESET_ALL} at ${price:.2f} (Total: ${revenue:.2f})")
        else:  # HOLD
            print(f"{Fore.BLUE + Style.BRIGHT}{action}:{Style.RESET_ALL} {quantity} shares of {Fore.YELLOW}{ticker}{Style.RESET_ALL}")
        
        # Print reasoning if available
        if 'reasoning' in transaction and transaction['reasoning']:
            print(f"{Fore.CYAN}Reasoning: {transaction['reasoning']}{Style.RESET_ALL}")
        
        print(Fore.BLUE + "-" * 60 + Style.RESET_ALL)

def print_game_summary(player, buffett, soros, stock_data):
    """Print end-of-game summary."""
    final_year = END_YEAR
    
    clear_screen()
    print_header()
    print("\n" + Fore.YELLOW + "=" * 80 + Style.RESET_ALL)
    print(Fore.YELLOW + Style.BRIGHT + " GAME OVER: FINAL RESULTS ".center(80, "*") + Style.RESET_ALL)
    print(Fore.YELLOW + "=" * 80 + Style.RESET_ALL + "\n")
    
    # Calculate final net worth for all players
    player_worth = player.calculate_net_worth({ticker: stock_data.loc[stock_data['Year'] == final_year, ticker].values[0] 
                                              for ticker in stock_data.columns if ticker != 'Year'})
    buffett_worth = buffett.calculate_net_worth({ticker: stock_data.loc[stock_data['Year'] == final_year, ticker].values[0] 
                                               for ticker in stock_data.columns if ticker != 'Year'})
    soros_worth = soros.calculate_net_worth({ticker: stock_data.loc[stock_data['Year'] == final_year, ticker].values[0] 
                                         for ticker in stock_data.columns if ticker != 'Year'})
    
    # Calculate total growth
    player_growth = (player_worth / INITIAL_CAPITAL - 1) * 100
    buffett_growth = (buffett_worth / INITIAL_CAPITAL - 1) * 100
    soros_growth = (soros_worth / INITIAL_CAPITAL - 1) * 100
    
    # Determine winner
    if player_worth >= buffett_worth and player_worth >= soros_worth:
        result = f"{Fore.GREEN + Style.BRIGHT}🏆 Congratulations! You beat the legendary investors! 🏆{Style.RESET_ALL}"
    elif player_worth >= buffett_worth or player_worth >= soros_worth:
        result = f"{Fore.CYAN + Style.BRIGHT}👍 Good job! You beat one of the legendary investors.{Style.RESET_ALL}"
    else:
        result = f"{Fore.RED}Better luck next time. The legends won this round.{Style.RESET_ALL}"
    
    print(result)
    print(f"\n{Fore.WHITE + Style.BRIGHT}Final Net Worth:{Style.RESET_ALL}")
    
    # Format player growth
    if player_growth > 0:
        player_growth_str = f"({Fore.GREEN}{player_growth:+.1f}%{Style.RESET_ALL})"
    else:
        player_growth_str = f"({Fore.RED}{player_growth:+.1f}%{Style.RESET_ALL})"
        
    # Format Buffett growth
    if buffett_growth > 0:
        buffett_growth_str = f"({Fore.GREEN}{buffett_growth:+.1f}%{Style.RESET_ALL})"
    else:
        buffett_growth_str = f"({Fore.RED}{buffett_growth:+.1f}%{Style.RESET_ALL})"
        
    # Format Soros growth
    if soros_growth > 0:
        soros_growth_str = f"({Fore.GREEN}{soros_growth:+.1f}%{Style.RESET_ALL})"
    else:
        soros_growth_str = f"({Fore.RED}{soros_growth:+.1f}%{Style.RESET_ALL})"
    
    print(f"{Fore.CYAN + Style.BRIGHT}You:{Style.RESET_ALL} ${player_worth:.2f} {player_growth_str}")
    print(f"{Fore.YELLOW}Warren Buffett:{Style.RESET_ALL} ${buffett_worth:.2f} {buffett_growth_str}")
    print(f"{Fore.MAGENTA}George Soros:{Style.RESET_ALL} ${soros_worth:.2f} {soros_growth_str}")
    
    print("\nPerformance Chart:")
    # Simple ASCII chart of net worth over time
    max_worth = max(max(player.net_worth_history), max(buffett.net_worth_history), max(soros.net_worth_history))
    chart_height = 10
    chart_width = END_YEAR - START_YEAR + 1
    
    print("\nNet Worth Over Time:")
    print(" " * 10 + "".join([f"{year:^8}" for year in range(START_YEAR, END_YEAR + 1)]))
    print(" " * 10 + "-" * (8 * chart_width))
    
    for y in range(chart_height, 0, -1):
        threshold = max_worth * y / chart_height
        line = f"${threshold:<8.0f} "
        
        for year_idx in range(chart_width):
            year_pos = year_idx + 1  # +1 because history starts with initial capital
            
            player_char = "*" if year_pos < len(player.net_worth_history) and player.net_worth_history[year_pos] >= threshold else " "
            buffett_char = "B" if year_pos < len(buffett.net_worth_history) and buffett.net_worth_history[year_pos] >= threshold else " "
            soros_char = "S" if year_pos < len(soros.net_worth_history) and soros.net_worth_history[year_pos] >= threshold else " "
            
            # Determine which character to display if there's overlap
            if player_char == "*" and buffett_char == "B" and soros_char == "S":
                display_char = "+"
            elif player_char == "*" and buffett_char == "B":
                display_char = "#"
            elif player_char == "*" and soros_char == "S":
                display_char = "@"
            elif buffett_char == "B" and soros_char == "S":
                display_char = "$"
            elif player_char == "*":
                display_char = "*"
            elif buffett_char == "B":
                display_char = "B"
            elif soros_char == "S":
                display_char = "S"
            else:
                display_char = " "
                
            line += f"{display_char:^8}"
            
        print(line)
    
    print(" " * 10 + "-" * (8 * chart_width))
    print("Legend: * = You, B = Buffett, S = Soros, # = You+Buffett, @ = You+Soros, $ = Buffett+Soros, + = All")
    
    print("\nThank you for playing Stock Market Showdown!")

def get_player_decision(player, available_stocks, stock_data, current_year):
    """Get decision input from the player."""
    while True:
        print(f"\n{Fore.CYAN + Style.BRIGHT}Available actions:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}1. Buy stocks{Style.RESET_ALL}")
        print(f"{Fore.RED}2. Sell stocks{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}3. View current stock prices{Style.RESET_ALL}")
        print(f"{Fore.BLUE}4. View portfolio{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}5. View AI investors' decisions{Style.RESET_ALL}")
        print(f"{Fore.WHITE}6. Get risk-seeking investment advice{Style.RESET_ALL}")
        print(f"{Fore.WHITE + Style.BRIGHT}7. Finish year{Style.RESET_ALL}")
        
        choice = input(f"\n{Fore.CYAN}Enter your choice (1-7): {Style.RESET_ALL}")
        
        if choice == '1':  # Buy
            print("\nAvailable stocks to buy:")
            # List all stocks with current prices
            for i, ticker in enumerate(available_stocks, 1):
                price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                print(f"{i}. {ticker:<6} ${price:.2f}")
            
            stock_idx = input("\nEnter stock number to buy (or 0 to cancel): ")
            if stock_idx == '0':
                continue
                
            try:
                stock_idx = int(stock_idx) - 1
                if stock_idx < 0 or stock_idx >= len(available_stocks):
                    print("Invalid stock number. Please try again.")
                    continue
                    
                ticker = available_stocks[stock_idx]
                price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                
                max_shares = int(player.cash / price)
                if max_shares <= 0:
                    print(f"You don't have enough cash to buy {ticker}.")
                    continue
                    
                print(f"\nYou have ${player.cash:.2f} cash.")
                print(f"Current price of {ticker}: ${price:.2f}")
                print(f"You can buy up to {max_shares} shares.")
                
                quantity = input(f"How many shares of {ticker} do you want to buy? (0 to cancel): ")
                if quantity == '0':
                    continue
                    
                try:
                    quantity = int(quantity)
                    if quantity <= 0:
                        continue
                        
                    if quantity > max_shares:
                        print(f"You can only afford {max_shares} shares. Buying {max_shares} instead.")
                        quantity = max_shares
                        
                    reasoning = input("Why are you buying this stock? (optional): ")
                    
                    success = player.buy(ticker, quantity, price, current_year, reasoning)
                    if success:
                        print(f"Successfully bought {quantity} shares of {ticker} for ${quantity * price:.2f}")
                    else:
                        print("Transaction failed.")
                except ValueError:
                    print("Please enter a valid number.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif choice == '2':  # Sell
            if not player.portfolio:
                print("You don't have any stocks to sell.")
                continue
                
            print("\nYour stocks to sell:")
            stocks_to_sell = []
            for i, (ticker, quantity) in enumerate(player.portfolio.items(), 1):
                price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                print(f"{i}. {ticker:<6} {quantity} shares at ${price:.2f} (Total: ${quantity * price:.2f})")
                stocks_to_sell.append(ticker)
                
            stock_idx = input("\nEnter stock number to sell (or 0 to cancel): ")
            if stock_idx == '0':
                continue
                
            try:
                stock_idx = int(stock_idx) - 1
                if stock_idx < 0 or stock_idx >= len(stocks_to_sell):
                    print("Invalid stock number. Please try again.")
                    continue
                    
                ticker = stocks_to_sell[stock_idx]
                quantity = player.portfolio[ticker]
                price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                
                sell_quantity = input(f"How many shares of {ticker} do you want to sell? (0 to cancel, max {quantity}): ")
                if sell_quantity == '0':
                    continue
                    
                try:
                    sell_quantity = int(sell_quantity)
                    if sell_quantity <= 0:
                        continue
                        
                    if sell_quantity > quantity:
                        print(f"You only have {quantity} shares. Selling all.")
                        sell_quantity = quantity
                        
                    reasoning = input("Why are you selling this stock? (optional): ")
                    
                    success = player.sell(ticker, sell_quantity, price, current_year, reasoning)
                    if success:
                        print(f"Successfully sold {sell_quantity} shares of {ticker} for ${sell_quantity * price:.2f}")
                    else:
                        print("Transaction failed.")
                except ValueError:
                    print("Please enter a valid number.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif choice == '3':  # View stock prices
            print("\nCurrent Stock Prices:")
            print(f"{'Ticker':<8} {'Price':<10} {'1Y Change':<12}")
            print("-" * 30)
            
            for ticker in available_stocks:
                current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                
                # Calculate price change if not first year
                if current_year > START_YEAR:
                    prev_price = stock_data.loc[stock_data['Year'] == current_year-1, ticker].values[0]
                    change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                    change_str = f"{change_pct:+.1f}%"
                else:
                    change_str = "N/A"
                    
                print(f"{ticker:<8} ${current_price:<9.2f} {change_str:<12}")
                
            input("\nPress Enter to continue...")
            
        elif choice == '4':  # View portfolio
            clear_screen()
            print_header()
            print_portfolio(player, stock_data, current_year)
            input("\nPress Enter to continue...")
            
        elif choice == '5':  # View AI decisions
            # Use current player portfolio stocks to simulate the AI investors' decisions
            # This makes it more relevant to what the player actually owns
            available_tickers = list(player.portfolio.keys()) + random.sample([t for t in available_stocks if t not in player.portfolio], min(5, len(available_stocks)))
            
            # Create temp portfolio for demo purposes (isolated from the actual game)
            temp_portfolio = {'cash': player.cash}
            for ticker in available_tickers:
                temp_portfolio[ticker] = player.portfolio.get(ticker, 0)
                
            market_event = market_event_fallback(current_year)
            
            print(f"\n{Fore.YELLOW + Style.BRIGHT}Warren Buffett's Approach:{Style.RESET_ALL}")
            print(Fore.YELLOW + "-" * 60 + Style.RESET_ALL)
            buffett_decisions = get_buffett_decisions(temp_portfolio, stock_data, current_year, market_event)
            
            if not buffett_decisions:
                print(f"{Fore.YELLOW}Warren Buffett has no recommendations at this time.{Style.RESET_ALL}")
            else:
                for decision in buffett_decisions:
                    ticker = decision.get('ticker', '')
                    action = decision.get('action', 'hold').upper()
                    quantity = decision.get('quantity', 0)
                    reasoning = decision.get('reasoning', 'No reasoning provided.')
                    
                    # Format based on action
                    if action == 'BUY':
                        print(f"{Fore.GREEN + Style.BRIGHT}{action}:{Style.RESET_ALL} {quantity} shares of {Fore.YELLOW}{ticker}{Style.RESET_ALL}")
                    elif action == 'SELL':
                        print(f"{Fore.RED + Style.BRIGHT}{action}:{Style.RESET_ALL} {quantity} shares of {Fore.YELLOW}{ticker}{Style.RESET_ALL}")
                    else:  # HOLD
                        print(f"{Fore.BLUE + Style.BRIGHT}{action}:{Style.RESET_ALL} {Fore.YELLOW}{ticker}{Style.RESET_ALL}")
                    
                    print(f"{Fore.CYAN}Reasoning: {reasoning}{Style.RESET_ALL}")
                    print()
                
            print(f"\n{Fore.MAGENTA + Style.BRIGHT}George Soros's Approach:{Style.RESET_ALL}")
            print(Fore.MAGENTA + "-" * 60 + Style.RESET_ALL)
            soros_decisions = get_soros_decisions(temp_portfolio, stock_data, current_year, market_event)
            
            if not soros_decisions:
                print(f"{Fore.MAGENTA}George Soros has no recommendations at this time.{Style.RESET_ALL}")
            else:
                for decision in soros_decisions:
                    ticker = decision.get('ticker', '')
                    action = decision.get('action', 'hold').upper()
                    quantity = decision.get('quantity', 0)
                    reasoning = decision.get('reasoning', 'No reasoning provided.')
                    
                    # Format based on action
                    if action == 'BUY':
                        print(f"{Fore.GREEN + Style.BRIGHT}{action}:{Style.RESET_ALL} {quantity} shares of {Fore.YELLOW}{ticker}{Style.RESET_ALL}")
                    elif action == 'SELL':
                        print(f"{Fore.RED + Style.BRIGHT}{action}:{Style.RESET_ALL} {quantity} shares of {Fore.YELLOW}{ticker}{Style.RESET_ALL}")
                    else:  # HOLD
                        print(f"{Fore.BLUE + Style.BRIGHT}{action}:{Style.RESET_ALL} {Fore.YELLOW}{ticker}{Style.RESET_ALL}")
                    
                    print(f"{Fore.CYAN}Reasoning: {reasoning}{Style.RESET_ALL}")
                    print()
                
            # Also offer recommendations on other available stocks they might consider
            print(f"\n{Fore.YELLOW + Style.BRIGHT}Warren Buffett might also consider:{Style.RESET_ALL}")
            potential_stocks = [s for s in available_stocks if s not in player.portfolio]
            if potential_stocks:
                recommended = random.sample(potential_stocks, min(3, len(potential_stocks)))
                for ticker in recommended:
                    current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                    # Generate Buffett-like commentary
                    buffett_advice = random.choice([
                        f"I like {ticker} at this price. It has the characteristics of a business with a durable competitive advantage.",
                        f"With {ticker}, I'm looking for businesses I can understand, with favorable long-term prospects, operated by honest and competent people, and available at attractive prices.",
                        f"{ticker} appears to have a strong moat. I prefer wonderful companies at fair prices rather than fair companies at wonderful prices.",
                        f"For {ticker}, I'd ask: would I buy the whole company at this price if I could? That's how I think about stocks."
                    ])
                    print(f"{Fore.YELLOW}• {ticker} (${current_price:.2f}): {buffett_advice}{Style.RESET_ALL}")
            
            print(f"\n{Fore.MAGENTA + Style.BRIGHT}George Soros might also consider:{Style.RESET_ALL}")
            if potential_stocks:
                recommended = random.sample(potential_stocks, min(3, len(potential_stocks)))
                for ticker in recommended:
                    current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                    # Generate Soros-like commentary
                    soros_advice = random.choice([
                        f"I see a potential reflexive dynamic developing with {ticker} where market perception and reality are creating a feedback loop.",
                        f"{ticker} presents an interesting case where market positioning might accelerate a trend once it becomes established.",
                        f"For {ticker}, I'm analyzing both the fundamentals and the market psychology to identify potential inflection points.",
                        f"With {ticker}, I'm watching for signs of market participants reinforcing a narrative that could create a self-fulfilling prophecy."
                    ])
                    print(f"{Fore.MAGENTA}• {ticker} (${current_price:.2f}): {soros_advice}{Style.RESET_ALL}")
                
            input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
            
        elif choice == '6':  # Get risk-seeking investment advice
            market_event = market_event_fallback(current_year)
            
            print(f"\n{Fore.CYAN + Style.BRIGHT}Risk-Seeking Investment Advice for {current_year}{Style.RESET_ALL}")
            print(Fore.CYAN + "-" * 70 + Style.RESET_ALL)
            
            # Get current market sentiment
            market_impact = sum(market_event['impact'].values()) / len(market_event['impact'])
            market_sentiment = "neutral"
            if market_impact < -3:
                market_sentiment = "extremely negative"
            elif market_impact < -1:
                market_sentiment = "negative"
            elif market_impact > 3:
                market_sentiment = "extremely positive"
            elif market_impact > 1:
                market_sentiment = "positive"
                
            # Current portfolio status
            total_invested = sum([stock_data.loc[stock_data['Year'] == current_year, ticker].values[0] * quantity 
                                for ticker, quantity in player.portfolio.items() if ticker != 'cash'])
            cash_ratio = player.cash / (player.cash + total_invested) if (player.cash + total_invested) > 0 else 1.0
            
            # Core advice based on risk profile
            print(f"{Fore.CYAN}Market Sentiment: {Style.BRIGHT}{market_sentiment.title()}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Current Cash Ratio: {Style.BRIGHT}{cash_ratio:.1%}{Style.RESET_ALL}")
            
            print(f"\n{Fore.WHITE + Style.BRIGHT}Core Strategy for Risk-Seeking Investors:{Style.RESET_ALL}")
            
            # Specific advice based on market conditions
            if market_sentiment == "extremely negative":
                print(f"• {Fore.RED}OPPORTUNITY:{Style.RESET_ALL} Market is in panic mode - the perfect time for contrarians.")
                print(f"• Consider using up to {Fore.GREEN}80-90%{Style.RESET_ALL} of your capital for strategic buys.")
                print(f"• Look for severely beaten-down tech and growth stocks with strong fundamentals.")
                print(f"• Focus on companies with sufficient cash to weather the storm.")
                print(f"• Remember: 'Be greedy when others are fearful' - Warren Buffett")
            elif market_sentiment == "negative":
                print(f"• {Fore.YELLOW}STRATEGY:{Style.RESET_ALL} Market decline presents buying opportunities for long-term gains.")
                print(f"• Consider allocating {Fore.GREEN}70-80%{Style.RESET_ALL} of your capital to high-conviction positions.")
                print(f"• Look for sectors showing relative strength despite overall market weakness.")
                print(f"• Consider averaging into high-quality growth stocks as they decline.")
            elif market_sentiment == "positive" or market_sentiment == "extremely positive":
                print(f"• {Fore.GREEN}MOMENTUM:{Style.RESET_ALL} Market is trending upward - ride the wave while staying vigilant.")
                print(f"• Maintain high equity exposure of {Fore.GREEN}80-90%{Style.RESET_ALL} to capture upside.")
                print(f"• Focus on sectors showing the strongest momentum and earnings growth.")
                print(f"• Consider using leverage or options for enhanced returns if you're experienced.")
                print(f"• Set trailing stops to protect profits while letting winners run.")
            else:  # neutral
                print(f"• Maintain an aggressive stance with {Fore.GREEN}70-80%{Style.RESET_ALL} in equities.")
                print(f"• Look for emerging trends and potential catalyst-driven opportunities.")
                print(f"• Consider rotation strategies to capture sector momentum shifts.")
                print(f"• Research potential disruptors that could see accelerating growth.")
            
            # Specific stock recommendations
            print(f"\n{Fore.WHITE + Style.BRIGHT}Suggested Actions for Your Portfolio:{Style.RESET_ALL}")
            
            # Review existing portfolio
            if player.portfolio:
                for ticker, quantity in player.portfolio.items():
                    if ticker == 'cash':
                        continue
                        
                    is_tech = any(keyword in ticker.lower() for keyword in ["tech", "net", "soft", "sys", "data", "micro", "com"])
                    is_financial = any(keyword in ticker.lower() for keyword in ["bank", "fin", "jp", "citi", "cap", "gs", "bac", "wfc"])
                    is_consumer = any(keyword in ticker.lower() for keyword in ["ko", "pg", "wmt", "mcd", "jnj", "pep"])
                    is_cyclical = any(keyword in ticker.lower() for keyword in ["auto", "steel", "retail", "energy", "oil", "gas"])
                    
                    if market_sentiment in ["extremely negative", "negative"]:
                        if is_tech:
                            print(f"• {Fore.GREEN}Consider increasing {ticker} position{Style.RESET_ALL} - tech stocks often lead recoveries after downturns.")
                        elif is_financial and market_sentiment == "extremely negative":
                            print(f"• {Fore.YELLOW}Watch {ticker} closely{Style.RESET_ALL} - financials could present opportunity once sentiment shifts.")
                        elif is_cyclical:
                            print(f"• {Fore.GREEN}Consider buying more {ticker}{Style.RESET_ALL} - cyclicals often outperform in early recovery phases.")
                        else:
                            print(f"• {Fore.YELLOW}Evaluate {ticker}'s growth potential{Style.RESET_ALL} against market recovery scenarios.")
                    else:
                        if is_tech:
                            print(f"• {Fore.GREEN}Focus on {ticker}{Style.RESET_ALL} - tech growth can outperform in bullish environments.")
                        elif is_cyclical:
                            print(f"• {Fore.GREEN}Monitor {ticker} for momentum{Style.RESET_ALL} - cyclicals often perform well in economic expansion.")
                        else:
                            print(f"• {Fore.YELLOW}Ensure {ticker} has sufficient growth catalysts{Style.RESET_ALL} to meet your return goals.")
            
            # Potential purchases
            print(f"\n{Fore.WHITE + Style.BRIGHT}High-Growth Investment Candidates:{Style.RESET_ALL}")
            growth_stocks = []
            tech_stocks = []
            momentum_stocks = []
            
            # Check for potential growth and tech stocks
            for ticker in available_stocks:
                # Identify growth stocks (using predetermined lists since we can't see actual historical data)
                if ticker in ["MSFT", "CSCO", "ORCL", "INTC", "IBM", "AAPL"]:  # Tech stocks from 2000
                    tech_stocks.append(ticker)
                elif ticker in ["AMZN", "EBAY", "YHOO"]:  # Early internet stocks
                    growth_stocks.append(ticker)
                
                # Track price changes to identify momentum stocks
                if current_year > START_YEAR:
                    current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                    prev_price = stock_data.loc[stock_data['Year'] == current_year-1, ticker].values[0]
                    change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                    
                    if change_pct > 15:  # Stocks with strong recent momentum
                        momentum_stocks.append((ticker, change_pct))
            
            # Show growth stock recommendations
            if tech_stocks:
                print(f"\n{Fore.CYAN}Tech Growth Opportunities:{Style.RESET_ALL}")
                for ticker in tech_stocks[:3]:  # Show top 3
                    current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                    print(f"• {Fore.GREEN}{ticker} (${current_price:.2f}){Style.RESET_ALL}: Technology company with potential for significant growth in the digital economy.")
            
            if growth_stocks:
                print(f"\n{Fore.MAGENTA}Emerging Growth Leaders:{Style.RESET_ALL}")
                for ticker in growth_stocks[:3]:  # Show top 3
                    current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                    print(f"• {Fore.GREEN}{ticker} (${current_price:.2f}){Style.RESET_ALL}: Emerging leader with disruptive business model and high growth potential.")
            
            # Show momentum stock recommendations
            if momentum_stocks:
                print(f"\n{Fore.YELLOW}Momentum Plays:{Style.RESET_ALL}")
                momentum_stocks.sort(key=lambda x: x[1], reverse=True)  # Sort by momentum
                for ticker, change_pct in momentum_stocks[:3]:  # Show top 3
                    current_price = stock_data.loc[stock_data['Year'] == current_year, ticker].values[0]
                    print(f"• {Fore.GREEN}{ticker} (${current_price:.2f}){Style.RESET_ALL}: Strong upward momentum with {change_pct:.1f}% recent gain.")
            
            # If no candidates found
            if not tech_stocks and not growth_stocks and not momentum_stocks:
                print(f"• {Fore.YELLOW}No clear high-growth candidates identified in current market conditions.{Style.RESET_ALL}")
                
            print(f"\n{Fore.WHITE + Style.BRIGHT}Remember:{Style.RESET_ALL} Risk-seeking investors prioritize capital appreciation over safety. The potential for outsized returns often comes with increased volatility and drawdowns.")
            
            input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
            
        elif choice == '7':  # Finish year
            # Mark any stocks not explicitly acted on as "hold"
            for ticker, quantity in player.portfolio.items():
                # Check if there was any action for this ticker this year
                if not any(t['ticker'] == ticker and t['year'] == current_year for t in player.history):
                    player.hold(ticker, current_year, "Decided to hold for now.")
            
            return
        
        else:
            print("Invalid choice. Please try again.")

# --- MAIN GAME LOOP ----------------------------------------------------------
def main():
    """Main game function."""
    # Initialize Agno agents
    initialize_agno_agents()
    
    clear_screen()
    print_header()
    print("\nWelcome to Stock Market Showdown!")
    print("You'll compete against Warren Buffett and George Soros in stock picking.")
    print(f"Starting in {START_YEAR} with ${INITIAL_CAPITAL:.2f}, your goal is to make the most money by {END_YEAR}.")
    print("\nEach year, you'll decide whether to buy, sell, or hold stocks.")
    print("Pay attention to market events and try to apply sound investment principles.")
    
    # Display user context information
    print(f"\n{Fore.CYAN + Style.BRIGHT}Your Investment Profile:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Risk Profile: {Style.BRIGHT}Risk Seeking{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Goal: {Style.BRIGHT}High Growth{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Strategy: {Style.BRIGHT}Pursue emerging trends and momentum opportunities{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Seek: {Style.BRIGHT}Growth stocks and disruptive technologies{Style.RESET_ALL}")
    
    print("\nGood luck!")
    
    # Get player name
    player_name = input("\nEnter your name: ")
    if not player_name:
        player_name = "Player"
    
    # Initialize players
    player = Player(player_name, INITIAL_CAPITAL)
    buffett = Player("Warren Buffett", INITIAL_CAPITAL)
    soros = Player("George Soros", INITIAL_CAPITAL)
    
    # Select 20 random stocks
    print("\nSelecting stocks for the game...")
    available_stocks = select_random_stocks()
    
    # Generate stock data
    print("Generating historical stock data...")
    stock_data = generate_stock_data(available_stocks)
    
    # Main game loop
    for year in range(START_YEAR, END_YEAR + 1):
        clear_screen()
        print_header()
        
        # Get market event for the year
        market_event = get_market_event(year)
        print_year_header(year, market_event)
        
        # Display portfolio and current status
        print_portfolio(player, stock_data, year)
        
        # Player makes decisions
        print("\nMake your investment decisions for this year.")
        get_player_decision(player, available_stocks, stock_data, year)
        
        # AI players make decisions
        print("\nWarren Buffett and George Soros are making their decisions...")
        
        # Prepare AI portfolios with cash and all available stocks (including ones they don't own)
        buffett_portfolio = {'cash': buffett.cash}
        for ticker in available_stocks:
            buffett_portfolio[ticker] = buffett.portfolio.get(ticker, 0)
            
        soros_portfolio = {'cash': soros.cash}
        for ticker in available_stocks:
            soros_portfolio[ticker] = soros.portfolio.get(ticker, 0)
        
        # Get AI decisions
        buffett_decisions = get_buffett_decisions(buffett_portfolio, stock_data, year, market_event)
        soros_decisions = get_soros_decisions(soros_portfolio, stock_data, year, market_event)
        
        # Execute AI decisions
        for decision in buffett_decisions:
            ticker = decision['ticker']
            action = decision['action']
            quantity = decision['quantity']
            price = stock_data.loc[stock_data['Year'] == year, ticker].values[0]
            reasoning = decision['reasoning']
            
            if action == 'buy':
                buffett.buy(ticker, quantity, price, year, reasoning)
            elif action == 'sell':
                buffett.sell(ticker, quantity, price, year, reasoning)
            else:  # hold
                if buffett.portfolio.get(ticker, 0) > 0:  # Only hold if we actually own the stock
                    buffett.hold(ticker, year, reasoning)
                
        for decision in soros_decisions:
            ticker = decision['ticker']
            action = decision['action']
            quantity = decision['quantity']
            price = stock_data.loc[stock_data['Year'] == year, ticker].values[0]
            reasoning = decision['reasoning']
            
            if action == 'buy':
                soros.buy(ticker, quantity, price, year, reasoning)
            elif action == 'sell':
                soros.sell(ticker, quantity, price, year, reasoning)
            else:  # hold
                if soros.portfolio.get(ticker, 0) > 0:  # Only hold if we actually own the stock
                    soros.hold(ticker, year, reasoning)
        
        # Update net worth for all players
        current_prices = {ticker: stock_data.loc[stock_data['Year'] == year, ticker].values[0] 
                         for ticker in stock_data.columns if ticker != 'Year'}
        player.update_net_worth_history(current_prices)
        buffett.update_net_worth_history(current_prices)
        soros.update_net_worth_history(current_prices)
        
        # Display year-end summary
        clear_screen()
        print_header()
        print_year_header(year, market_event)
        print("Year-End Summary:")
        print_decision_history(player, year)
        print_leaderboard(player, buffett, soros, year, stock_data)
        
        if year < END_YEAR:
            input("\nPress Enter to continue to the next year...")
    
    # Game over - display summary
    print_game_summary(player, buffett, soros, stock_data)

if __name__ == "__main__":
    main()