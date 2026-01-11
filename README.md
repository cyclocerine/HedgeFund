[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N51EF9I0)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)

# AI Hedge Fund V2.3

**Enterprise-grade Algorithmic Trading with P-LSTM & Hybrid Actor-Critic PPO**

This system represents a major architectural leap in autonomous trading, combining latest advances in Deep Learning (Time-Series Patching) and Reinforcement Learning (Memory-Aware Agents) to navigate complex financial markets.

<div align="center">
  <img src="/results/NVDA_20260108_234831_plot.png" alt="AI Hedge Fund Performance" width="800">
  <p><em>System Performance: Hybrid PPO Agent identifying trend reversal on NVDA</em></p>
</div>

---

## 🚀 Key Features (Version 2.3)

### 1. Vectorized Training Engine (New)
- **High-Performance Training**: 4x-8x training speedup using parallel environments.
- **Batched Inference**: Efficiently processes multiple market states simultaneously.
- **Isolated Data Streams**: Prevents data leakage by shuffling start times across environments.

### 2. Hybrid Actor-Critic Architecture (New)
- **Memory-Aware PPO**: Integrates an **LSTM Cell** into the PPO agent's decision core.
- **Transformer Encoder**: Uses a mini-PatchTST encoder for state feature extraction.
- **Deep Context**: Agent "remembers" past market regimes (bull/bear) via hidden states, enabling smarter long-term decisions.

### 3. P-LSTM Forecasting Model (New)
- **Patch-LSTM**: Breaks time-series into patches (e.g., length 16) to feed into LSTM.
- **Result**: Reduces sequence length by 16x, allowing the model to see much longer history (e.g., 512 days) without vanishing gradients.
- **Residual Connections**: ResNet-style skip connections for stable deep training.

### 4. Cross-Ticker Generalization (New)
- **Universal Agent**: One agent trained simultaneously on multiple assets (e.g., BTC + ETH + NVDA).
- **Macro-Awareness**: Integartes global macro signals (NASDAQ, DJI, TNX, VIX) to understand market sentiment.
- **Robustness**: Prevents overexpression to a single stock's specific price action.

---

## 🏗️ System Architecture

### 1. Data Pipeline
- **Input**: OHLCV data + Global Macro Indicators (VIX, TNX, Indices).
- **Feature Engineering**:
  - **Stationary Features**: Log returns, Z-Scores, Normalized Volume.
  - **Technical Signals**: MACD, RSI, Bollinger Bands, ADX.
  - **Macro Regime**: Risk-On/Risk-Off scoring based on bond yields.

### 2. Decision Engine (The Agent)
The `HybridActorCritic` module processes the market state:
1.  **Feature Encoder**: Raw features $\rightarrow$ Transformer Block $\rightarrow$ Latent Vector.
2.  **Memory Core**: Latent Vector + Previous Hidden State $\rightarrow$ LSTM Cell $\rightarrow$ New Hidden State.
3.  **Policy Head (Actor)**: Outputs action distribution (Buy/Sell/Hold).
4.  **Value Head (Critic)**: Estimates expected future reward (Portfolio Value).

### 3. Execution & Validation
- **Vectorized Backtesting**: Simulates years of trading in minutes.
- **Realistic Slippage**: Dynamic transaction costs based on volatility (ATR).
- **Walk-Forward Validation**: Strict separation of Train (2020-2024) and Test (2025-2026) periods.

---

## 🛠️ Usage Guide

### Prerequisities
```bash
pip install -r requirements.txt
```

### 1. Standard Prediction
Generate trading signals for a single stock using the latest model.
```bash
python scripts/run_cli.py --ticker NVDA --mode predict --ppo
```

### 2. High-Speed Training (Vectorized)
Train the PPO agent 4x faster using parallel environments.
```bash
python scripts/run_cli.py --ticker NVDA --mode predict --ppo --vectorized --n-envs 8
```

### 3. Hybrid Memory-Aware Agent
Enable the LSTM memory core for better temporal context.
```bash
python scripts/run_cli.py --ticker NVDA --mode predict --ppo --hybrid-ac
```

### 4. Cross-Ticker Training
Train a universal agent on multiple assets (e.g., Crypto & Tech).
```bash
# Train on BTC and ETH
python scripts/tune_ppo.py --mode train --tickers BTC-USD ETH-USD --episodes 100 --save-model saved_models/ppo/cross_crypto.pt

# Use the trained model for ETH prediction
# Note: Manually copy/rename model if needed for run_cli auto-load
copy saved_models/ppo/cross_crypto.pt saved_models/ppo/ETH-USD_enhanced_v2.3.pt
python scripts/run_cli.py --ticker ETH-USD --mode predict --ppo
```

---

## 📊 Performance Benchmark

**Validation Results (Jan 2026)**

| Metric | Standard PPO | Hybrid PPO (Memory) | Cross-Ticker Agent |
|:-------|:-------------|:--------------------|:-------------------|
| **Sharpe Ratio** | 1.0 - 1.5 | **1.8 - 2.5** | 1.2 - 2.0 |
| **Win Rate** | ~35-40% | **~45-50%** | ~40-45% |
| **Generalization** | Low (Specific) | Medium | **High (Universal)** |

> **Note**: Hybrid PPO shows superior risk-adjusted returns due to its ability to "wait out" volatile periods using memory.

---

## 🤝 Algorithms Explanation

### **Proximal Policy Optimization (PPO)**
Instead of learning a Q-value (DQN), PPO learns a **Policy** (probability of taking action given state). It clips updates to prevent drastic changes that could destabilize the agent—essential for financial time-series where data is noisy.

### **Patching (P-LSTM & PatchTST)**
Financial data is dense. Instead of processing day-by-day (t, t+1, t+2...), we group days into **Patches** (e.g., 2 weeks).
- **Benefit**: The model sees "shapes" (trends) rather than just points.
- **Result**: Semantic understanding of "Dip", "Rally", "Consolidation".

### **Curriculum Learning**
The agent isn't thrown into the deep end. We train it in phases:
1.  **Phase 1 (Easy)**: Low fees, no noise. Learn basic "Buy Low, Sell High".
2.  **Phase 2 (Medium)**: Realistic fees, moderate noise. Learn efficient execution.
3.  **Phase 3 (Hard)**: High fees, high noise (Chaos Mode). Learn risk management and survival.

---

## 📜 License
MIT License - Open for modification and commercial use.
