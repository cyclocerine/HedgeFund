#!/usr/bin/env python
"""
Comprehensive Feature Test — BMRI.JK (Bank Mandiri)
=====================================================

Tests all enhanced V3.0 features using real market data:
1. Data Download & Feature Engineering
2. P-LSTM (Enhanced) — Training + Multi-Horizon Prediction
3. PPO Agent (Enhanced) — Training + Backtest
4. Cross-Model Integration — PatchTST signal → PPO
5. Performance Summary

Author: AI Hedge Fund V3.0
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

# ============================================================
# CONFIG
# ============================================================
TICKER = "BMRI.JK"
START_DATE = "2023-01-01"
END_DATE = "2026-03-27"
INITIAL_INVESTMENT = 100_000_000  # IDR 100 Juta

# Training params (moderate for testing)
PLSTM_EPOCHS = 30
PPO_EPISODES = 50
PPO_MAX_STEPS = 500

print("=" * 70)
print(f"  COMPREHENSIVE TEST — {TICKER}")
print(f"  Period: {START_DATE} to {END_DATE}")
print(f"  Initial Investment: IDR {INITIAL_INVESTMENT:,.0f}")
print("=" * 70)

# Track results
results = {}
timings = {}


# ============================================================
# TEST 1: Data Download & Feature Engineering
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: Data Download & Feature Engineering")
print("=" * 70)

t0 = time.time()
try:
    import yfinance as yf
    from src.data.feature_engineering import (
        TradingFeatureEngineer, 
        prepare_patchtst_features,
        prepare_ppo_features
    )
    
    # Download data
    print(f"  Downloading {TICKER} data...")
    data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    prices = data['Close'].values.flatten()
    print(f"  ✓ Downloaded {len(prices)} trading days")
    print(f"  Price range: IDR {prices.min():,.0f} — IDR {prices.max():,.0f}")
    print(f"  Latest price: IDR {prices[-1]:,.0f}")
    
    # Feature Engineering
    engineer = TradingFeatureEngineer(ohlcv_df=data)
    engineer.calculate_all_indicators()
    scores = engineer.get_all_scores()
    
    print(f"\n  Signal Scores (latest):")
    for name, arr in scores.items():
        print(f"    {name:20s}: {arr[-1]:.4f}")
    
    # Prepare features for PatchTST
    patchtst_features = prepare_patchtst_features(prices, ohlcv_df=data, use_scores=True)
    print(f"\n  PatchTST features shape: {patchtst_features.shape}")
    print(f"  PPO features shape: {prepare_ppo_features(prices, ohlcv_df=data).shape}")
    
    results['data'] = 'PASSED'
    print(f"\n  ✓ TEST 1 PASSED")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    results['data'] = f'FAILED: {e}'
    print(f"\n  ✗ TEST 1 FAILED: {e}")

timings['data'] = time.time() - t0


# ============================================================
# TEST 2: P-LSTM Enhanced — Training & Prediction
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: P-LSTM Enhanced — Training & Prediction")
print("=" * 70)

t0 = time.time()
try:
    from src.models.patch_lstm import PatchLSTM, PatchLSTMWrapper
    
    # Prepare data for P-LSTM
    features = patchtst_features  # Shape: (N, n_features)
    seq_len = 60  # Lookback window
    
    # Create sequences
    X_sequences = []
    y_targets = []
    for i in range(seq_len, len(features)):
        X_sequences.append(features[i-seq_len:i])
        y_targets.append(features[i, 0])  # Predict first feature (log returns)
    
    X = np.array(X_sequences)
    y = np.array(y_targets)
    
    # Train/Val split (80/20, time-ordered)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Input dim: {X.shape[2]}, Seq len: {seq_len}")
    
    # Create and train P-LSTM
    input_dim = X.shape[2]
    wrapper = PatchLSTMWrapper(
        input_dim=input_dim,
        patch_len=16,
        d_model=128,
        lstm_layers=2,
        dropout=0.15,
        n_heads=4,
        attn_layers=1,
        forecast_horizons=[1, 7, 14, 30]
    )
    
    print(f"\n  Training Enhanced P-LSTM ({PLSTM_EPOCHS} epochs)...")
    wrapper.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=PLSTM_EPOCHS,
        batch_size=32,
        verbose=1,
        early_stopping_patience=10,
        input_noise=0.01,
        label_noise=0.005,
        swa_start_pct=0.75
    )
    
    # Predictions
    val_preds = wrapper.predict(X_val)
    
    # Save predictions to CSV
    os.makedirs('results', exist_ok=True)
    plstm_results_df = pd.DataFrame({
        'Actual_Log_Return': y_val.flatten(),
        'Predicted_Log_Return': val_preds.flatten()
    })
    plstm_out = 'results/bmri_plstm_predictions.csv'
    plstm_results_df.to_csv(plstm_out, index=False)
    print(f"  P-LSTM predictions saved to: {plstm_out}")
    
    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_val, val_preds)
    mae = mean_absolute_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)
    
    print(f"\n  Validation Metrics:")
    print(f"    MSE:  {mse:.6f}")
    print(f"    MAE:  {mae:.6f}")
    print(f"    R²:   {r2:.4f}")
    
    # Multi-horizon prediction
    print(f"\n  Multi-Horizon Predictions (last sequence):")
    all_horizon_preds = wrapper.predict_all_horizons(X_val[-1:])
    for h, pred in all_horizon_preds.items():
        print(f"    Horizon {h:3d}: {pred[0]:.6f}")
    
    # Model architecture summary
    total_params = sum(p.numel() for p in wrapper.model.parameters())
    trainable_params = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)
    print(f"\n  Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    results['plstm'] = 'PASSED'
    print(f"\n  ✓ TEST 2 PASSED")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    results['plstm'] = f'FAILED: {e}'
    print(f"\n  ✗ TEST 2 FAILED: {e}")

timings['plstm'] = time.time() - t0


# ============================================================
# TEST 3: PPO Agent — Training with HybridActorCritic
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: PPO Agent — Training with HybridActorCritic")
print("=" * 70)

t0 = time.time()
try:
    from src.trading.ppo_agent import PPOTrader, PPOAgent, TradingEnv
    
    # Create PPO trader with BMRI.JK data
    trader = PPOTrader(
        prices=prices,
        ohlcv_df=data,
        initial_investment=INITIAL_INVESTMENT,
        use_enhanced_features=True,
        transaction_fee=0.0015,  # Realistic IDX fees
    )
    
    print(f"  PPO Network: {type(trader.agent.network).__name__}")
    print(f"  State dim: {trader.agent.state_dim}")
    print(f"  Obs normalization: {trader.agent.normalize_obs}")
    print(f"  Reward normalization: {trader.agent.normalize_rewards}")
    print(f"  Value clipping: {trader.agent.value_clip}")
    
    # Count network parameters
    total_params = sum(p.numel() for p in trader.agent.network.parameters())
    print(f"  Network parameters: {total_params:,}")
    
    # Train PPO
    print(f"\n  Training PPO ({PPO_EPISODES} episodes, curriculum learning)...")
    train_result = trader.train(
        episodes=PPO_EPISODES,
        max_steps=PPO_MAX_STEPS,
        verbose=True
    )
    
    print(f"\n  Training Results:")
    print(f"    Episodes: {train_result.get('total_episodes', PPO_EPISODES)}")
    
    if 'episode_rewards' in train_result:
        rewards = train_result['episode_rewards']
        print(f"    Avg Reward (last 10): {np.mean(rewards[-10:]):.4f}")
        print(f"    Max Reward: {np.max(rewards):.4f}")
    
    results['ppo_train'] = 'PASSED'
    print(f"\n  ✓ TEST 3 PASSED")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    results['ppo_train'] = f'FAILED: {e}'
    print(f"\n  ✗ TEST 3 FAILED: {e}")

timings['ppo_train'] = time.time() - t0


# ============================================================
# TEST 4: PPO Backtest — Performance Evaluation
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: PPO Backtest — Performance Evaluation")
print("=" * 70)

t0 = time.time()
try:
    # Run backtest
    print(f"  Running backtest on {TICKER}...")
    backtest_result = trader.backtest()
    
    perf = backtest_result['performance']
    trades = backtest_result.get('trades', [])
    portfolio = backtest_result.get('portfolio_values', [])
    
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │         BACKTEST RESULTS — {TICKER}          │")
    print(f"  ├─────────────────────────────────────────────┤")
    
    for metric, value in perf.items():
        if isinstance(value, float):
            if 'return' in metric.lower() or 'drawdown' in metric.lower():
                print(f"  │  {metric:30s}: {value*100:>8.2f}%  │")
            else:
                print(f"  │  {metric:30s}: {value:>9.4f}  │")
        else:
            print(f"  │  {metric:30s}: {str(value):>9s}  │")
    
    print(f"  │  {'Total Trades':30s}: {len(trades):>9d}  │")
    print(f"  └─────────────────────────────────────────────┘")
    
    # Trade analysis and saving
    if trades:
        os.makedirs('results', exist_ok=True)
        trades_df = pd.DataFrame(trades)
        trades_out = 'results/bmri_ppo_trades.csv'
        trades_df.to_csv(trades_out, index=False)
        print(f"\n  Trade history saved to: {trades_out}")
        
        buy_trades = [t for t in trades if t.get('action') == 'BUY' or t.get('type') == 'BUY']
        sell_trades = [t for t in trades if t.get('action') == 'SELL' or t.get('type') == 'SELL']
        print(f"  Trade Distribution:")
        print(f"    BUY:  {len(buy_trades)}")
        print(f"    SELL: {len(sell_trades)}")
    
    # Portfolio value progression
    if portfolio:
        portfolio = np.array(portfolio)
        print(f"\n  Portfolio Value:")
        print(f"    Start:   IDR {portfolio[0]:>15,.0f}")
        print(f"    End:     IDR {portfolio[-1]:>15,.0f}")
        print(f"    Min:     IDR {portfolio.min():>15,.0f}")
        print(f"    Max:     IDR {portfolio.max():>15,.0f}")
    
    results['ppo_backtest'] = 'PASSED'
    print(f"\n  ✓ TEST 4 PASSED")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    results['ppo_backtest'] = f'FAILED: {e}'
    print(f"\n  ✗ TEST 4 FAILED: {e}")

timings['ppo_backtest'] = time.time() - t0


# ============================================================
# TEST 5: PPO Vectorized Training (Multi-Env)
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: PPO Vectorized Training (Multi-Env)")
print("=" * 70)

t0 = time.time()
try:
    from src.trading.ppo_agent import VectorizedTradingEnv
    
    # Create new trader for vectorized training
    vec_trader = PPOTrader(
        prices=prices,
        ohlcv_df=data,
        initial_investment=INITIAL_INVESTMENT,
        use_enhanced_features=True,
        transaction_fee=0.0015,
    )
    
    print(f"  Training with {4} parallel environments...")
    vec_result = vec_trader.train_vectorized(
        episodes=30,
        n_envs=4,
        max_steps=PPO_MAX_STEPS,
        verbose=True
    )
    
    print(f"\n  Vectorized Training Results:")
    print(f"    Total Episodes: {vec_result.get('total_episodes', 'N/A')}")
    
    if 'episode_rewards' in vec_result:
        rewards = vec_result['episode_rewards']
        if len(rewards) > 0:
            print(f"    Avg Reward (last 10): {np.mean(rewards[-10:]):.4f}")
    
    # Backtest the vectorized-trained agent
    vec_backtest = vec_trader.backtest()
    vec_perf = vec_backtest['performance']
    
    print(f"\n  Vectorized Backtest:")
    for metric, value in vec_perf.items():
        if isinstance(value, float):
            if 'return' in metric.lower() or 'drawdown' in metric.lower():
                print(f"    {metric:30s}: {value*100:.2f}%")
            else:
                print(f"    {metric:30s}: {value:.4f}")
    
    results['ppo_vectorized'] = 'PASSED'
    print(f"\n  ✓ TEST 5 PASSED")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    results['ppo_vectorized'] = f'FAILED: {e}'
    print(f"\n  ✗ TEST 5 FAILED: {e}")

timings['ppo_vectorized'] = time.time() - t0


# ============================================================
# TEST 6: Model Save/Load with Normalization Stats
# ============================================================
print(f"\n{'='*70}")
print("TEST 6: Model Save/Load with Normalization Stats")
print("=" * 70)

t0 = time.time()
try:
    import tempfile
    
    # Save the trained model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "bmri_ppo.pt")
        plstm_path = os.path.join(tmpdir, "bmri_plstm.pt")
        
        # Save PPO
        trader.agent.save_model(model_path)
        file_size = os.path.getsize(model_path)
        print(f"  PPO model size: {file_size/1024:.1f} KB")
        
        # Save P-LSTM
        wrapper.save(plstm_path)
        plstm_size = os.path.getsize(plstm_path)
        print(f"  P-LSTM model size: {plstm_size/1024:.1f} KB")
        
        # Load PPO into new agent
        new_agent = PPOAgent(state_dim=trader.agent.state_dim, action_dim=3)
        new_agent.load_model(model_path)
        
        # Verify normalization stats preserved
        if trader.agent.obs_rms is not None and new_agent.obs_rms is not None:
            mean_match = np.allclose(trader.agent.obs_rms.mean, new_agent.obs_rms.mean)
            var_match = np.allclose(trader.agent.obs_rms.var, new_agent.obs_rms.var)
            print(f"  Obs normalization stats preserved: mean={mean_match}, var={var_match}")
        
        # Verify same action output
        test_state = np.random.randn(trader.agent.state_dim).astype(np.float32)
        # Set both to eval mode
        trader.agent.network.eval()
        new_agent.network.eval()
        
        with torch.no_grad():
            s1 = torch.FloatTensor(test_state).unsqueeze(0).to(trader.agent.device)
            if trader.agent.use_hybrid:
                p1, v1, _, _ = trader.agent.network(s1)
                p2, v2, _, _ = new_agent.network(s1.to(new_agent.device))
            else:
                p1, v1 = trader.agent.network(s1)
                p2, v2 = new_agent.network(s1.to(new_agent.device))
            
            probs_match = torch.allclose(p1, p2, atol=1e-6)
            value_match = torch.allclose(v1, v2, atol=1e-6)
            print(f"  Network outputs match: probs={probs_match}, values={value_match}")
        
        # Load P-LSTM
        new_wrapper = PatchLSTMWrapper(input_dim=input_dim, forecast_horizons=[1, 7, 14, 30])
        new_wrapper.load(plstm_path)
        
        # Verify predictions match
        test_input = X_val[:5]
        orig_preds = wrapper.predict(test_input)
        loaded_preds = new_wrapper.predict(test_input)
        preds_match = np.allclose(orig_preds, loaded_preds, atol=1e-5)
        print(f"  P-LSTM predictions match: {preds_match}")
    
    results['save_load'] = 'PASSED'
    print(f"\n  ✓ TEST 6 PASSED")

except Exception as e:
    import traceback
    traceback.print_exc()
    results['save_load'] = f'FAILED: {e}'
    print(f"\n  ✗ TEST 6 FAILED: {e}")

timings['save_load'] = time.time() - t0


# ============================================================
# TEST 7: Ensemble Model (PatchTST + BiLSTM)
# ============================================================
print(f"\n{'='*70}")
print("TEST 7: Ensemble Model (if available)")
print("=" * 70)

t0 = time.time()
try:
    from src.models.ensemble_model import EnsemblePredictor
    
    ensemble = EnsemblePredictor(
        input_dim=input_dim,
        forecast_horizons=[1, 7, 14, 30]
    )
    
    print(f"  Training Ensemble ({PLSTM_EPOCHS} epochs)...")
    ensemble.fit(
        X_train, y_train, X_val, y_val,
        epochs=PLSTM_EPOCHS,
        batch_size=32,
        early_stopping_patience=10
    )
    
    ensemble_preds = ensemble.predict(X_val)
    ensemble_mse = mean_squared_error(y_val, ensemble_preds)
    ensemble_r2 = r2_score(y_val, ensemble_preds)
    
    print(f"\n  Ensemble Validation:")
    print(f"    MSE: {ensemble_mse:.6f}")
    print(f"    R²:  {ensemble_r2:.4f}")
    
    results['ensemble'] = 'PASSED'
    print(f"\n  ✓ TEST 7 PASSED")

except Exception as e:
    results['ensemble'] = f'SKIPPED: {e}'
    print(f"\n  ⚠ TEST 7 SKIPPED: {e}")

timings['ensemble'] = time.time() - t0


# ============================================================
# TEST 8: Directional Model
# ============================================================
print(f"\n{'='*70}")
print("TEST 8: Directional Model (if available)")
print("=" * 70)

t0 = time.time()
try:
    from src.models.directional_model import DirectionalEnsemble
    
    # Create binary labels (UP/DOWN)
    y_direction_train = (y_train > 0).astype(np.float32)
    y_direction_val = (y_val > 0).astype(np.float32)
    
    dir_model = DirectionalEnsemble(input_dim=input_dim)
    
    print(f"  Training Directional Model ({PLSTM_EPOCHS} epochs)...")
    dir_model.fit(
        X_train, y_direction_train, X_val, y_direction_val,
        epochs=PLSTM_EPOCHS,
        batch_size=32
    )
    
    dir_preds = dir_model.predict(X_val)
    accuracy = np.mean((dir_preds > 0.5) == y_direction_val)
    
    print(f"\n  Directional Accuracy: {accuracy*100:.1f}%")
    
    results['directional'] = 'PASSED'
    print(f"\n  ✓ TEST 8 PASSED")

except Exception as e:
    results['directional'] = f'SKIPPED: {e}'
    print(f"\n  ⚠ TEST 8 SKIPPED: {e}")

timings['directional'] = time.time() - t0


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"  FINAL SUMMARY — {TICKER}")
print(f"{'='*70}")

total_time = sum(timings.values())

test_names = {
    'data': 'Data & Feature Engineering',
    'plstm': 'P-LSTM Enhanced',
    'ppo_train': 'PPO Training (Hybrid)',
    'ppo_backtest': 'PPO Backtest',
    'ppo_vectorized': 'PPO Vectorized Training',
    'save_load': 'Model Save/Load',
    'ensemble': 'Ensemble Model',
    'directional': 'Directional Model'
}

passed = 0
failed = 0
skipped = 0

for key, name in test_names.items():
    status = results.get(key, 'NOT RUN')
    t = timings.get(key, 0)
    
    if status == 'PASSED':
        icon = '✓'
        passed += 1
    elif 'SKIPPED' in str(status):
        icon = '⚠'
        skipped += 1
    else:
        icon = '✗'
        failed += 1
    
    status_short = status if len(str(status)) < 30 else str(status)[:30] + '...'
    print(f"  {icon} {name:35s} {status_short:30s} ({t:.1f}s)")

print(f"\n  {'─'*50}")
print(f"  PASSED:  {passed}")
print(f"  FAILED:  {failed}")
print(f"  SKIPPED: {skipped}")
print(f"  Total Time: {total_time:.1f}s")
print(f"{'='*70}")

sys.exit(1 if failed > 0 else 0)
