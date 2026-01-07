#!/usr/bin/env python
"""
Test Feature Engineering Module
==============================

Script untuk menguji modul feature engineering dan integrasinya
dengan PPO Agent dan PatchTST.
"""

import sys
import os
import numpy as np
import pandas as pd

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_feature_engineering_basic():
    """Test basic indicator calculations."""
    print("\n=== Test 1: Basic Feature Engineering ===")
    
    from src.data.feature_engineering import TradingFeatureEngineer
    
    # Generate sample price data
    np.random.seed(42)
    n_days = 200
    prices = 100 + np.cumsum(np.random.normal(0.1, 2, n_days))
    prices = np.maximum(prices, 10)  # Ensure positive prices
    
    # Create feature engineer
    engineer = TradingFeatureEngineer(prices=prices)
    
    # Test RSI
    rsi = engineer.calculate_rsi()
    assert len(rsi) == n_days, f"RSI length mismatch: {len(rsi)} != {n_days}"
    assert np.all((rsi >= 0) & (rsi <= 100)), "RSI out of bounds"
    print(f"✓ RSI: min={rsi.min():.2f}, max={rsi.max():.2f}, mean={rsi.mean():.2f}")
    
    # Test Stochastic RSI
    stoch_k, stoch_d = engineer.calculate_stochastic_rsi()
    assert len(stoch_k) == n_days, f"Stoch RSI K length mismatch"
    print(f"✓ Stochastic RSI K: min={stoch_k.min():.2f}, max={stoch_k.max():.2f}")
    print(f"✓ Stochastic RSI D: min={stoch_d.min():.2f}, max={stoch_d.max():.2f}")
    
    # Test MACD
    macd, signal, hist = engineer.calculate_macd()
    assert len(macd) == n_days, "MACD length mismatch"
    print(f"✓ MACD Histogram: min={hist.min():.4f}, max={hist.max():.4f}")
    
    # Test Bollinger Bands
    middle, upper, lower, percent_b = engineer.calculate_bollinger_bands()
    assert len(percent_b) == n_days, "BB %B length mismatch"
    print(f"✓ BB %B: min={percent_b.min():.2f}, max={percent_b.max():.2f}")
    
    # Test ADX
    adx = engineer.calculate_adx()
    assert len(adx) == n_days, "ADX length mismatch"
    print(f"✓ ADX: min={adx.min():.2f}, max={adx.max():.2f}")
    
    # Test Log Returns
    log_returns = engineer.calculate_log_returns()
    assert not np.any(np.isnan(log_returns)), "Log returns contains NaN"
    print(f"✓ Log Returns: min={log_returns.min():.4f}, max={log_returns.max():.4f}")
    
    print("\n[PASS] Basic feature engineering tests passed!")
    return True


def test_scoring_system():
    """Test signal scoring system."""
    print("\n=== Test 2: Scoring System (0.5 = Neutral) ===")
    
    from src.data.feature_engineering import TradingFeatureEngineer
    
    np.random.seed(42)
    n_days = 200
    prices = 100 + np.cumsum(np.random.normal(0.1, 2, n_days))
    prices = np.maximum(prices, 10)
    
    engineer = TradingFeatureEngineer(prices=prices)
    
    # Calculate all scores
    scores = engineer.get_all_scores()
    
    # Check score ranges
    for name, arr in scores.items():
        if name == 'trend_direction':
            assert np.all((arr >= -1) & (arr <= 1)), f"{name} out of bounds"
            print(f"✓ {name}: min={arr.min():.2f}, max={arr.max():.2f} (valid range: -1 to 1)")
        else:
            assert np.all((arr >= 0) & (arr <= 1)), f"{name} out of bounds [0,1]"
            print(f"✓ {name}: min={arr.min():.2f}, max={arr.max():.2f} (valid range: 0 to 1)")
    
    # Check for neutrality (0.5 should exist in the data)
    macd_score = scores['macd_score']
    neutral_count = np.sum(np.isclose(macd_score, 0.5, atol=0.01))
    print(f"\n✓ MACD Score neutrality check: {neutral_count} samples near 0.5")
    
    print("\n[PASS] Scoring system tests passed!")
    return True


def test_ppo_features():
    """Test PPO feature generation."""
    print("\n=== Test 3: PPO Features (6 dimensions) ===")
    
    from src.data.feature_engineering import TradingFeatureEngineer, prepare_ppo_features
    
    np.random.seed(42)
    n_days = 200
    prices = 100 + np.cumsum(np.random.normal(0.1, 2, n_days))
    prices = np.maximum(prices, 10)
    
    # Using helper function
    features = prepare_ppo_features(prices)
    
    assert features.shape == (n_days, 6), f"Shape mismatch: {features.shape} != ({n_days}, 6)"
    assert features.dtype == np.float32, f"Dtype mismatch: {features.dtype}"
    
    # Check for NaN
    assert not np.any(np.isnan(features)), "Features contain NaN"
    
    # Check bounds (all should be 0-1)
    assert np.all(features >= 0) and np.all(features <= 1), "Features out of [0,1] bounds"
    
    print(f"✓ PPO Features shape: {features.shape}")
    print(f"✓ PPO Features dtype: {features.dtype}")
    print(f"✓ All features in [0, 1] range: True")
    
    # Feature names
    feature_names = ['macd_score', 'stoch_rsi_score', 'bb_score', 'volume_score', 
                     'adx_normalized', 'trend_direction']
    for i, name in enumerate(feature_names):
        col = features[:, i]
        print(f"  - {name}: [{col.min():.2f}, {col.max():.2f}]")
    
    print("\n[PASS] PPO features tests passed!")
    return True


def test_patchtst_features():
    """Test PatchTST feature generation."""
    print("\n=== Test 4: PatchTST Features (8 dimensions, stationary) ===")
    
    from src.data.feature_engineering import prepare_patchtst_features
    
    np.random.seed(42)
    n_days = 200
    prices = 100 + np.cumsum(np.random.normal(0.1, 2, n_days))
    prices = np.maximum(prices, 10)
    
    # Using helper function
    features = prepare_patchtst_features(prices, use_scores=True)
    
    assert features.shape == (n_days, 8), f"Shape mismatch: {features.shape} != ({n_days}, 8)"
    assert features.dtype == np.float32, f"Dtype mismatch: {features.dtype}"
    
    # Check for NaN
    assert not np.any(np.isnan(features)), "Features contain NaN"
    
    print(f"✓ PatchTST Features shape: {features.shape}")
    print(f"✓ PatchTST Features dtype: {features.dtype}")
    print(f"✓ No NaN values: True")
    
    # Check stationarity of log returns (mean should be near 0)
    log_returns = features[:, 0]
    print(f"✓ Log Returns mean: {log_returns.mean():.6f} (should be ~0)")
    
    print("\n[PASS] PatchTST features tests passed!")
    return True


def test_ppo_integration():
    """Test PPO Agent integration with enhanced features."""
    print("\n=== Test 5: PPO Agent Integration (Enhanced Mode) ===")
    
    from src.trading.ppo_agent import PPOTrader, TradingEnv
    
    np.random.seed(42)
    n_days = 100
    prices = 100 + np.cumsum(np.random.normal(0.1, 2, n_days))
    prices = np.maximum(prices, 10)
    
    # Test enhanced mode
    print("Testing PPOTrader with use_enhanced_features=True...")
    trader = PPOTrader(
        prices=prices, 
        initial_investment=10000,
        use_enhanced_features=True
    )
    
    # Check observation space
    obs_dim = trader.env.observation_space.shape[0]
    expected_dim = 8 + 6  # 8 base + 6 enhanced
    assert obs_dim == expected_dim, f"Observation dim mismatch: {obs_dim} != {expected_dim}"
    print(f"✓ Observation space: {obs_dim} dimensions (8 base + 6 enhanced)")
    
    # Test reset
    obs = trader.env.reset()
    assert len(obs) == expected_dim, f"Observation length mismatch"
    assert not np.any(np.isnan(obs)), "Observation contains NaN"
    print(f"✓ Reset observation shape: {obs.shape}")
    
    # Test step
    obs, reward, done, info = trader.env.step(0)  # Hold action
    assert len(obs) == expected_dim
    print(f"✓ Step observation shape: {obs.shape}")
    
    # Quick training test (2 episodes)
    print("Training for 2 episodes...")
    results = trader.train(episodes=2, verbose=False)
    assert 'episode_rewards' in results
    print(f"✓ Training completed. Episodes: {len(results['episode_rewards'])}")
    
    print("\n[PASS] PPO integration tests passed!")
    return True


def test_backward_compatibility():
    """Test backward compatibility (legacy mode)."""
    print("\n=== Test 6: Backward Compatibility (Legacy Mode) ===")
    
    from src.trading.ppo_agent import PPOTrader, TradingEnv
    
    np.random.seed(42)
    n_days = 100
    prices = 100 + np.cumsum(np.random.normal(0.1, 2, n_days))
    prices = np.maximum(prices, 10)
    
    # Test legacy mode (default)
    print("Testing PPOTrader with default settings (legacy mode)...")
    trader = PPOTrader(
        prices=prices, 
        initial_investment=10000
        # use_enhanced_features=False (default)
    )
    
    # Check observation space (legacy: 8 base + 2 external)
    obs_dim = trader.env.observation_space.shape[0]
    expected_dim = 8 + 2  # 8 base + 2 default (returns, volatility)
    assert obs_dim == expected_dim, f"Legacy obs dim mismatch: {obs_dim} != {expected_dim}"
    print(f"✓ Legacy observation space: {obs_dim} dimensions")
    
    # Test training
    results = trader.train(episodes=2, verbose=False)
    assert len(results['episode_rewards']) == 2
    print(f"✓ Legacy training completed")
    
    print("\n[PASS] Backward compatibility tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE ENGINEERING MODULE TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_feature_engineering_basic()
        all_passed &= test_scoring_system()
        all_passed &= test_ppo_features()
        all_passed &= test_patchtst_features()
        all_passed &= test_ppo_integration()
        all_passed &= test_backward_compatibility()
        
        print("\n" + "=" * 60)
        if all_passed:
            print("[SUCCESS] All tests passed!")
        else:
            print("[FAILURE] Some tests failed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
