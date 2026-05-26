[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N51EF9I0)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)

# AI Hedge Fund V3.0

**Enterprise-grade Algorithmic Trading with P-LSTM, Continuous Action Space, & Hybrid Risk Guardrails**

This system represents a major architectural leap in autonomous trading, combining the latest advances in Deep Learning (Time-Series Patching), Reinforcement Learning (Continuous PPO), and institutional-grade Quantitative Risk Management.

<div align="center">
  <img src="/results/NVDA_20260108_234831_plot.png" alt="AI Hedge Fund Performance" width="800">
  <p><em>System Performance: Hybrid PPO Agent identifying trend reversal on NVDA</em></p>
</div>

---

##  Key Features (Version 3.0)

### 1. Continuous Action Space (New)
- **Granular Control**: PPO now outputs continuous actions `[-1.0, 1.0]` instead of discrete Buy/Sell/Hold, enabling precise position sizing (e.g., "Invest 65% of capital").
- **Gaussian Distribution**: The Actor network uses `mu` and `log_std` to explore the action space more naturally, allowing for smoother equity curves.

### 2. Hybrid Risk Guardrail (New)
- **Hard Ceiling**: Prevents "Black Swan" all-in behaviors by enforcing a maximum position size per trade (e.g., 50% max allocation).
- **Volatility Scaling**: Dynamically tightens position limits when the Average True Range (ATR) indicates high market turbulence.
- **Drawdown Circuit Breaker**: Automatically slashes the agent's trading size limit if portfolio drawdown exceeds safe thresholds (e.g., cuts sizing by 90% at 15% drawdown).

### 3. Indonesian Macroeconomic Data Integration (New)
- **Bank-Specific Indicators**: Custom data pipeline for Indonesian big-cap stocks (e.g., BBNI.JK) integrating:
  - **Liquidity & Currency**: USD/IDR, IHSG (^JKSE).
  - **Foreign Flow Proxies**: EIDO (iShares MSCI Indonesia), EEM (MSCI Emerging Markets).
  - **Commodities**: Crude Oil WTI, Gold Futures.
- **Log Returns**: Automatically applies log-return transformations to macro data for better stationarity.

### 4. Advanced Out-of-Sample (OOS) Backtester (New)
- **Standalone Engine**: A dedicated script (`static_backtest.py`) to strictly separate In-Sample training (e.g., 2010-2020) and Out-of-Sample testing (e.g., 2020-2026) to prevent data leakage.
- **Dynamic Slippage**: Simulates real-world market impact by scaling slippage based on local ATR.
- **IDX Tick Size Validation**: Rounds execution prices to comply with the official Indonesia Stock Exchange (IDX) tick size rules.

---

##  System Architecture

### 1. Data Pipeline
- **Input**: OHLCV data + Global/Local Macro Indicators.
- **Feature Engineering**:
  - **Stationary Features**: Log returns, Z-Scores, Normalized Volume.
  - **Technical Signals**: MACD, RSI, Bollinger Bands, ADX, ATR.

### 2. Decision Engine (The Agent)
The `HybridActorCritic` module processes the market state:
1.  **Feature Encoder**: Raw features $\rightarrow$ Transformer Block $\rightarrow$ Latent Vector.
2.  **Memory Core**: Latent Vector + Previous Hidden State $\rightarrow$ LSTM Cell $\rightarrow$ New Hidden State.
3.  **Policy Head (Actor)**: Outputs a continuous action mean ($\mu$) and standard deviation ($\sigma$).
4.  **Value Head (Critic)**: Estimates expected future reward (Portfolio Value).
5.  **Risk Manager**: Clamps the Actor's output based on real-time portfolio health and market volatility.

### 3. Execution & Validation
- **OOS Testing**: Prevents concept drift illusions by testing exclusively on unseen market regimes.
- **Walk-Forward Validation**: Supports rolling window retraining for continuous adaptation.

---

##  Usage Guide

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Data Preparation (Indonesian Stocks + Macro)
Download stock data along with crucial macroeconomic indicators.
```bash
python scripts/download_bbni_data.py
```

### 2. Out-of-Sample Static Backtesting
Run a strict OOS backtest with continuous actions and dynamic slippage.
```bash
python scripts/static_backtest.py --csv data/BBNI_JK.csv --capital 100000000 --epochs 50 --output oos_backtest.png
```

### 3. Standard Prediction (Auto-Macro Integrated)
Generate trading signals using the latest model. The system will **automatically download** and merge 10 global/local macroeconomic indicators (USD/IDR, IHSG, EIDO, EEM, Oil, Gold, TNX, etc.) alongside the requested ticker for high-accuracy forecasting.
```bash
# Example for Indonesian Bank Stock
python scripts/run_cli.py --ticker BMRI.JK --mode predict --ppo --hybrid-ac --force-retrain

# Example for US Tech Stock
python scripts/run_cli.py --ticker NVDA --mode predict --ppo --hybrid-ac
```

### 4. High-Speed Training (Vectorized)
Train the PPO agent faster using parallel environments.
```bash
python scripts/run_cli.py --ticker NVDA --mode predict --ppo --vectorized --n-envs 8
```

### 5. CLI Flags Explanation
Here is the complete breakdown of the command used for Auto-Macro Prediction:
`python scripts/run_cli.py --ticker BMRI.JK --mode predict --ppo --hybrid-ac --force-retrain`

1. **`--ticker BMRI.JK`**
   - **Function**: Specifies the target stock ticker symbol.
   - **Context**: In this example, it fetches *Bank Mandiri* (`BMRI.JK`) from the Indonesia Stock Exchange. If replaced with `AAPL` or `BTC-USD`, the model automatically adapts to fetch the most relevant macro data for that asset.

2. **`--mode predict`**
   - **Function**: Runs the full "Forecasting" pipeline.
   - **Context**: The system doesn't just stop at training; it proceeds to forecast prices for the next 30 days. If you use `--mode train` instead, the system only learns, saves its weights (the `.pt` file), and stops.

3. **`--ppo`**
   - **Function**: Activates the Reinforcement Learning execution agent.
   - **Context**: The deep learning model (PatchTST) predicts prices, but ignores fees or capital constraints. The `--ppo` flag summons the RL Trading Agent to translate those price predictions into a final, continuous action (e.g., 0% to 100% Buy/Sell allocation).

4. **`--hybrid-ac`**
   - **Function**: Upgrades the PPO agent's brain to be *Memory-Aware*.
   - **Context**: It fuses the Actor-Critic PPO network with an LSTM memory cell. Without this, the agent makes decisions based purely on today's state. With `--hybrid-ac`, the agent remembers past market regimes, e.g., "The trend has been bearish all month, I should hold back on buying even if the price spiked today."

5. **`--force-retrain`**
   - **Function**: Forces the system to train from scratch, overwriting old checkpoints.
   - **Context**: Financial markets are highly dynamic (*concept drift*). A model trained last month might fail to grasp today's market sentiment. This flag ensures the AI thoroughly re-evaluates its weights to capture the latest market patterns before trading.

---

##  Algorithms Explanation

### **Continuous PPO with Guardrails**
Unlike discrete reinforcement learning, our Continuous PPO agent doesn't just decide *what* to do, but *how much* to do. However, RL agents are notoriously greedy in unseen states. Our Hybrid Risk Guardrail intercepts the agent's raw signal (e.g., `0.9` or 90% capital) and mathematically clamps it down to a safer size (e.g., `0.3`) if volatility is spiking or a drawdown is occurring.

### **Patching (P-LSTM & PatchTST)**
Financial data is dense. Instead of processing day-by-day (t, t+1, t+2...), we group days into **Patches** (e.g., 2 weeks).
- **Benefit**: The model sees "shapes" (trends) rather than just points.
- **Result**: Semantic understanding of "Dip", "Rally", "Consolidation".

---

##  License
MIT License - Open for modification and commercial use.
