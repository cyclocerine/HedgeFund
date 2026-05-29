# AI Hedge Fund - Comprehensive System Documentation

## 1. Introduction
The AI Hedge Fund is an enterprise-grade algorithmic trading system that leverages state-of-the-art Deep Learning (DL) for time-series forecasting and Reinforcement Learning (RL) for automated trade execution. The system is designed to handle multi-asset portfolios, ingest global and local macroeconomic data, and operate under strict quantitative risk management guardrails.

---

## 2. System Architecture

The system is built upon a hybrid architecture consisting of three main pillars:

### A. Deep Learning Engine (The "Predictor")
- **Models**: Uses advanced architectures like **PatchTST** (Transformer-based) and **P-LSTM** (Serial Cascade Patch-LSTM).
- **Mechanism**: Instead of feeding data day-by-day (which causes vanishing gradients over long horizons), the models use "Patching". The time-series data is divided into sub-sequences (patches) allowing the model to learn semantic shapes (e.g., bull flags, head and shoulders) over extended histories (up to 512 days lookback).
- **Output**: Predicts future price trajectories (e.g., next 30 days).

### B. Reinforcement Learning Engine (The "Trader")
- **Model**: **Continuous Proximal Policy Optimization (PPO)**.
- **Mechanism**: While the DL engine predicts where the price will go, the RL agent decides *what to do* about it. It operates in a continuous action space `[-1.0, 1.0]`, allowing for granular position sizing rather than binary Buy/Sell decisions.
- **Hybrid Actor-Critic**: The PPO agent is infused with an LSTM memory cell. This gives the agent temporal context (memory), allowing it to recognize market regimes (e.g., avoiding aggressive buys during a sustained bear market).

### C. Quantitative Risk Management (The "Guardrail")
- RL agents can be unpredictable in out-of-distribution (Black Swan) events. The system employs a `RiskManager` that acts as a hard ceiling.
- It dynamically scales down the PPO agent's requested position size based on real-time portfolio drawdown, asset volatility (ATR), and maximum allocation limits.

---

## 3. Data Pipeline

The data pipeline is designed to prevent data leakage and provide a rich multivariate feature set to the AI.

1. **Ingestion**: Fetches OHLCV (Open, High, Low, Close, Volume) data via `yfinance`.
2. **Macroeconomic Integration**: Automatically downloads crucial global and local macro indicators (e.g., USD/IDR, Jakarta Composite Index, EIDO, Crude Oil, Gold, 10-Year Treasury Yield).
3. **Feature Engineering**: 
   - Converts non-stationary prices into **Log Returns**.
   - Calculates momentum and volatility technical indicators (MACD, RSI, Bollinger Bands, ATR, ADX).
   - Generates normalized signal scores (0.0 to 1.0) and regime identifiers.
4. **Environment Wrapping**: Wraps the processed data into an OpenAI Gym-compatible environment (`TradingEnv`) for the RL agent to interact with.

---

## 4. Key Directories and Functions

### `src/data/` (Data Processing Layer)
- **`data_loader.py` / `preprocessor.py`**: Handles downloading stock data and orchestrating feature generation.
- **`feature_engineering.py`**: 
  - `fetch_macro_data()`: Connects to Yahoo Finance to pull global/local indices and commodities.
  - `prepare_macro_features()`: Merges macro data with the primary stock data, handling forward-filling for timezone/holiday mismatches.
  - `get_features_for_patchtst()`: Packages the multivariate stationary features for the deep learning models.

### `src/models/` (Deep Learning Layer)
- **`patch_lstm.py`**: Implements the `PatchLSTM` architecture, bridging patch-based processing with recurrent memory cells.
- **`patchtst.py`**: Implements the channel-independent Patch Time Series Transformer.

### `src/trading/` (Trading & RL Layer)
- **`ppo_agent.py`**: Contains `PPOAgent`, the RL execution engine. It handles environment vectorization, rollout collection, and policy updates.
- **`trading_env.py`**: Custom Gym environment simulating the stock market. It calculates rewards based on the agent's actions and current portfolio value.
- **`risk_manager.py`**: 
  - `clamp_ppo_action()`: Intercepts the raw continuous action from the PPO agent and mathematically reduces the position size if volatility is too high or drawdown limits are breached.
- **`backtest.py`**: The core simulation engine for executing historical trades, calculating slippage, and generating equity curves.

### `scripts/` (Execution Layer)
- **`run_cli.py`**: The main command-line interface. Connects the data loader, DL model, and PPO agent into a single unified pipeline for training and prediction.
- **`static_backtest.py`**: A strict Out-of-Sample (OOS) testing environment. Separates training data (e.g., 2010-2020) from testing data (2020-2026) to prove the model's validity against concept drift.
- **`download_bbni_data.py`**: A specialized script for downloading Indonesian equity data alongside relevant macro indicators.

---

## 5. Usage Guide

### A. Environment Setup
Ensure you have Python 3.10+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### B. Standard Forecasting & Trading Signal Generation
Use the main CLI to generate predictions. The system will automatically download the stock data and merge it with macro indicators.
```bash
python scripts/run_cli.py --ticker NVDA --mode predict --ppo --hybrid-ac
```
- `--ticker`: The target asset (e.g., `NVDA`, `BMRI.JK`, `BTC-USD`).
- `--mode predict`: Generates future signals. Use `--mode train` to only train the weights.
- `--ppo`: Activates the Reinforcement Learning agent to output continuous Buy/Sell allocation percentages.
- `--hybrid-ac`: Enables LSTM memory inside the PPO agent for better regime awareness.
- `--force-retrain` *(optional)*: Bypasses cached models and trains from scratch using the newest data.

### C. High-Speed Vectorized Training
If you want to train the RL agent much faster, utilize multiple parallel CPU environments:
```bash
python scripts/run_cli.py --ticker AAPL --mode predict --ppo --vectorized --n-envs 8
```

### D. Out-of-Sample Backtesting
To rigorously test the system's performance on unseen data (with dynamic slippage and continuous actions), use the static backtester.
First, prepare a dataset:
```bash
python scripts/download_bbni_data.py
```
Then run the backtest:
```bash
python scripts/static_backtest.py --csv data/BBNI_JK.csv --capital 100000000 --epochs 50 --output oos_backtest.png
```

---

## 6. Conclusion
The AI Hedge Fund merges institutional-level risk management with cutting-edge Deep Learning and Reinforcement Learning. Its automated macroeconomic data pipeline allows it to contextually understand market forces across multiple asset classes, making it a robust and adaptable automated trading system.
