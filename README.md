# EdTech Simulator Collection

This repository contains a collection of educational simulator applications focused on finance and business education.

## Stock Market Showdown

A CLI game where you compete against Warren Buffett and Cathie Wood in stock picking.

### Features:
- Start in the year 2000 with $10,000
- Choose from 20 random stocks over 10 years (2000-2010)
- Compete against AI agents mimicking Warren Buffett and Cathie Wood
- Make investment decisions based on historical market events
- Learn investment strategies from legendary investors

### Installation:
```bash
pip install -r requirements.txt
```

### Running the game:
```bash
python stock_market_game.py
```

Requires OpenAI API key set as an environment variable or in a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Other Simulators

### Portfolio Analyzer
A Streamlit app for learning asset allocation and compound interest.

```bash
streamlit run portfolio_sim_app.py
```

### Marketing Launch Simulator
Streamlit prototype featuring 3 AI agents to simulate marketing decisions.

```bash
streamlit run marketing_sim_app.py
```

### Portfolio Game Simulator
A game-like interactive financial simulator for learning asset allocation, compounding, and behavioral finance.

```bash
streamlit run portfolio_game_app.py
```

## Requirements

See `requirements.txt` for dependencies.