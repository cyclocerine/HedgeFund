[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N51EF9I0)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)

# AI Hedge Fund

**Advanced Algorithmic Trading System with PatchTST & PPO Reinforcement Learning**

An enterprise-grade portfolio management and automated trading system leveraging state-of-the-art Deep Learning (**PatchTST**) and Reinforcement Learning (**PPO**) to navigate financial markets.

<div align="center">
  <img src="/results/NVDA_20260108_234831_plot.png" alt="AI Hedge Fund Performance" width="800">
  <p><em>System Performance: Backtest on NVDA showing +107% Return</em></p>
</div>

---

## Key Features

### 1. State-of-the-art Forecasting (PatchTST)
- **Transformer Architecture**: Utilizes the latest PatchTST model for time-series forecasting
- **Multi-Horizon Prediction**: Forecasts 1, 7, 14, and 30 days ahead simultaneously
- **Bayesian Hyperparameter Tuning**: Integrated Optuna optimization for model fine-tuning

### 2. Intelligent Trading Agent (PPO v2.2)
- **Reinforcement Learning**: Uses Proximal Policy Optimization (PPO) for autonomous trading decisions
- **Curriculum Learning**: 3-phase training (Easy → Medium → Hard) for robust agent development
- **Asymmetric Reward Scaling**: `reward * 15` with `1.2x` bonus for profitable trades
- **Entropy Decay**: Dynamic exploration-exploitation balance (0.05 → 0.001)
- **Enhanced Features**: 14+ technical signals including MACD, Stochastic RSI, Bollinger Bands, ADX

### 3. Robustness Framework
- **Stress Testing**: Automated testing under high fees (1%), noise injection (2%), and normal conditions
- **Softmax-based Signals**: Trading signals derived directly from model's probability distribution
- **Calibrated Risk Management**: EMA-200 trend filter, dynamic stop-loss, drawdown penalties

### 4. Unified Validation System
- **Aligned Modes**: Predict and Backtest modes share exact same feature engineering and agent logic
- **Walk-Forward Validation**: Train on 2020-2024, test on 2025-2026
- **Performance Metrics**: Automatic calculation of Sharpe Ratio, Max Drawdown, Win Rate, and Total Return

---

## Latest Performance (January 2026)

### Stress Test Results on NVDA (High-Alpha Market)

| Scenario | Fee | Return | Sharpe | MDD | Win Rate |
|:---------|:----|:-------|:-------|:----|:---------|
| **Baseline (Normal)** | 0.1% | **+76.68%** | **1.00** | 10.44% | 37.8% |
| **High Stress** | 1.0% | **+78.58%** | **1.08** | 9.50% | 29.4% |
| **Chaos Mode (Noise)** | 0.1% | **+107.31%** | **1.17** | 10.70% | 33.1% |

> **All scenarios profitable with Sharpe > 1.0!**

### Comparison: Low-Alpha vs High-Alpha Markets

| Metric | BMRI.JK (Sideways) | NVDA (Volatile) |
|:-------|:-------------------|:----------------|
| Baseline Return | -3.27% | **+76.68%** |
| Sharpe Ratio | -0.45 | **+1.00** |
| Max Drawdown | 3.62% | 10.44% |

> **Insight**: Bot performs best on high-alpha assets with clear trends.

---

## Installation

```bash
# Clone repository
git clone https://github.com/cyclocerine/HedgeFund.git
cd HedgeFund

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage Guide

The unified CLI (`scripts/run_cli.py`) is the main entry point for all operations.

### 1. Predict Future Prices & Generate Signals

```bash
# Basic Prediction with PPO Signals
python scripts/run_cli.py --ticker NVDA --mode predict --ppo --ppo-episodes 100 --save-results

# With PatchTST Hyperparameter Tuning (Recommended)
python scripts/run_cli.py --ticker NVDA --mode predict --tune --ppo --ppo-episodes 200 --forecast-days 30 --save-results
```

### 2. Backtest Trading Strategy

```bash
# Run PPO Backtest (200 Episodes)
python scripts/run_cli.py --ticker NVDA --mode backtest --strategy PPO --tune --ppo-episodes 200 --initial-balance 100000000 --save-results
```

### 3. Run Stress Test

```bash
# Validate agent robustness under extreme conditions
python scripts/stress_test.py
```

### CLI Arguments

| Argument | Description | Default |
|:---------|:------------|:--------|
| `--ticker` | Asset symbol (e.g., NVDA, BMRI.JK, TSLA) | Required |
| `--mode` | `predict` or `backtest` | Required |
| `--strategy` | Trading strategy (`PPO`, `Trend Following`, `Mean Reversion`) | `PPO` |
| `--tune` | Enable PatchTST hyperparameter tuning | `False` |
| `--ppo` | Enable PPO trading signals | `False` |
| `--ppo-episodes` | Number of training episodes for PPO | `200` |
| `--train-noise` | Noise injection level during training (0.0-0.1) | `0.0` |
| `--forecast-days` | Days to forecast into the future | `30` |
| `--initial-balance` | Starting capital for backtest | `100,000,000` |
| `--save-results` | Save plots and CSVs to `results/` | `False` |

---

## Technical Deep Dive

### PPO Agent Architecture (v2.2)

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO AGENT V2.2                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Easy Phase │ -> │ Medium Phase│ -> │  Hard Phase │     │
│  │  Fee: 0.1%  │    │  Fee: 0.2%  │    │  Fee: 0.5%  │     │
│  │  Noise: 0%  │    │  Noise: 1%  │    │  Noise: 2%  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ↓                  ↓                  ↓             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           CURRICULUM LEARNING (3-Phase)             │   │
│  │   Episodes: 0-33% -> 33-66% -> 66-100%              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    REWARD SHAPING                           │
├─────────────────────────────────────────────────────────────┤
│  reward = log(ratio) * 15           # Asymmetric Scaling   │
│  if profit: reward *= 1.2           # Bonus for winning    │
│  reward -= delta_drawdown * 12      # Risk penalty         │
│  if action != HOLD: reward -= fee   # Transaction cost     │
│  if action != HOLD && profit:       # Activity bonus       │
│      reward += 0.05                                         │
└─────────────────────────────────────────────────────────────┘
```

### Signal Generation (Softmax-based)

```python
# Instead of historical action counting (50% confidence bug)
# We now query the trained network directly:

action_probs, _ = ppo_trader.agent.network(state_tensor)
probs = action_probs.cpu().numpy().flatten()

# probs[0] = HOLD, probs[1] = BUY, probs[2] = SELL
if buy_prob > 0.5:
    signal = 'BUY', confidence = buy_prob * 100
elif sell_prob > 0.5:
    signal = 'SELL', confidence = sell_prob * 100
else:
    signal = 'HOLD', confidence = hold_prob * 100
```

### Enhanced Feature Engineering

| Feature Category | Signals Included |
|:-----------------|:-----------------|
| **Trend** | EMA-200, Price Momentum (10/20/50 day) |
| **Momentum** | RSI, Stochastic RSI, MACD Histogram |
| **Volatility** | Bollinger Band Position, ATR, Historical Volatility |
| **Volume** | Volume Ratio, OBV Momentum, VWAP Distance |
| **Trend Strength** | ADX Score (0.0-1.0 normalized) |

---

## System Architecture

```
+-------------------+       +----------------------+
|    Market Data    | ----> |     Data Pipeline    |
|  (yfinance API)   |       |   (OHLCV + Volume)   |
+-------------------+       +----------------------+
                                       |
                                       v
                            +----------------------+
                            |  Feature Engineering |
                            |  (14+ Technical      |
                            |   Indicators)        |
                            +----------------------+
                                       |
                  +--------------------+--------------------+
                  |                                         |
                  v                                         v
        +------------------+                      +------------------+
        |  PatchTST Model  |                      |   PPO Agent V2.2 |
        |  (Transformer)   |                      |  (Curriculum +   |
        |                  |                      |   Asymmetric)    |
        +------------------+                      +------------------+
                  |                                         |
                  v                                         v
        +------------------+                      +------------------+
        | Price Prediction |                      |  Trading Signals |
        | (1/7/14/30 days) |                      |  (Softmax-based) |
        +------------------+                      +------------------+
                  |                                         |
                  +--------------------+--------------------+
                                       |
                                       v
                            +----------------------+
                            |    Stress Testing    |
                            |  (Fee/Noise/Chaos)   |
                            +----------------------+
                                       |
                                       v
                            +----------------------+
                            |  Portfolio Execution |
                            +----------------------+
```

---

## Project Structure

```
HedgeFund/
├── scripts/
│   ├── run_cli.py           # Main CLI entry point
│   └── stress_test.py       # Robustness validation
├── src/
│   ├── data/
│   │   └── feature_engineering.py   # Technical indicators
│   ├── models/
│   │   ├── patchtst_model.py        # PatchTST implementation
│   │   └── predictor.py             # Stock predictor wrapper
│   └── trading/
│       ├── ppo_agent.py             # PPO Agent V2.2
│       ├── backtest.py              # Backtesting engine
│       └── optimizer.py             # Strategy optimizer
├── results/                 # Output plots and CSVs
├── requirements.txt
└── README.md
```

---

## Contributing

Contributions are welcome! Key files to explore:

| File | Description |
|:-----|:------------|
| `src/trading/ppo_agent.py` | PPO Agent with Curriculum Learning & Asymmetric Rewards |
| `src/models/patchtst_model.py` | Transformer-based Price Forecasting |
| `src/data/feature_engineering.py` | Technical Indicator Processing |
| `scripts/stress_test.py` | Robustness Validation Framework |

---

## References

- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- De Prado, M. L. (2018). *Advances in Financial Machine Learning*
- Murphy, J. J. *Technical Analysis of the Financial Markets*
- Nie, Y., et al. (2022). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)*
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*

---

## License

[MIT](LICENSE)

---

<div align="center">
  <p><strong>AI Hedge Fund V2.2</strong> - Intelligent Algorithmic Trading for the Digital Era</p>
  <p><em>Built with PyTorch, PPO, and PatchTST</em></p>
</div>
