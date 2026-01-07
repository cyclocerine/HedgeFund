#!/usr/bin/env python
"""
Test Model dengan Saham BBCA
============================

Script untuk menguji PPO Agent dan PatchTST dengan data saham BBCA.
Menggunakan enhanced features dari TradingFeatureEngineer.
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def download_bbca_data(period="2y"):
    """Download data BBCA dari Yahoo Finance."""
    print(f"\n📊 Downloading BBCA.JK data ({period})...")
    
    ticker = yf.Ticker("BBCA.JK")
    df = ticker.history(period=period)
    
    if df.empty:
        raise ValueError("Failed to download BBCA data")
    
    print(f"✓ Downloaded {len(df)} days of data")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Price range: Rp {df['Close'].min():,.0f} - Rp {df['Close'].max():,.0f}")
    
    return df


def test_feature_engineering_bbca(df):
    """Test feature engineering dengan data BBCA."""
    print("\n" + "="*60)
    print("TEST 1: Feature Engineering dengan BBCA")
    print("="*60)
    
    from src.data.feature_engineering import TradingFeatureEngineer
    
    # Create engineer with OHLCV data
    engineer = TradingFeatureEngineer(ohlcv_df=df)
    
    # Calculate all indicators
    engineer.calculate_all_indicators()
    
    # Get scores
    scores = engineer.get_all_scores()
    
    print("\n📈 Signal Scores (last 5 days):")
    print("-" * 50)
    for name, values in scores.items():
        last_val = values[-1]
        interpretation = "Neutral"
        if name == 'trend_direction':
            if last_val > 0.5: interpretation = "Bullish"
            elif last_val < -0.5: interpretation = "Bearish"
            else: interpretation = "Sideways"
        else:
            if last_val > 0.7: interpretation = "Bullish"
            elif last_val < 0.3: interpretation = "Bearish"
            else: interpretation = "Neutral"
        print(f"  {name:20s}: {last_val:.3f} ({interpretation})")
    
    # Get features for PPO
    ppo_features = engineer.get_features_for_ppo_all()
    print(f"\n✓ PPO Features shape: {ppo_features.shape}")
    
    # Get features for PatchTST
    tst_features = engineer.get_features_for_patchtst()
    print(f"✓ PatchTST Features shape: {tst_features.shape}")
    
    return engineer, scores


def test_ppo_bbca(df, episodes=20):
    """Test PPO Agent dengan data BBCA."""
    print("\n" + "="*60)
    print("TEST 2: PPO Agent dengan BBCA (Enhanced Features)")
    print("="*60)
    
    from src.trading.ppo_agent import PPOTrader
    
    prices = df['Close'].values
    
    # Create PPO Trader with enhanced features
    print("\n🤖 Initializing PPO Agent...")
    trader = PPOTrader(
        prices=prices,
        initial_investment=100_000_000,  # 100 juta
        use_enhanced_features=True,
        ohlcv_df=df
    )
    
    print(f"✓ State dimensions: {trader.env.observation_space.shape[0]}")
    print(f"✓ Action space: {trader.env.action_space.n} (Hold, Buy, Sell)")
    
    # Train
    print(f"\n🚀 Training for {episodes} episodes...")
    results = trader.train(episodes=episodes, verbose=True)
    
    # Backtest
    print("\n📊 Running backtest...")
    backtest = trader.backtest()
    
    perf = backtest['performance']
    print("\n" + "-"*50)
    print("📈 BACKTEST RESULTS")
    print("-"*50)
    print(f"  Initial Investment : Rp {perf['initial_investment']:>15,.0f}")
    print(f"  Final Value        : Rp {perf['final_value']:>15,.0f}")
    print(f"  Total Return       : {perf['total_return']:>14.2f}%")
    print(f"  Max Drawdown       : {perf['max_drawdown']:>14.2f}%")
    print(f"  Sharpe Ratio       : {perf['sharpe_ratio']:>14.4f}")
    print(f"  Win Rate           : {perf['win_rate']:>14.2f}%")
    print(f"  Number of Trades   : {perf['num_trades']:>14d}")
    
    return trader, backtest


def test_ppo_comparison(df, episodes=20):
    """Compare PPO with and without enhanced features."""
    print("\n" + "="*60)
    print("TEST 3: A/B Comparison (Enhanced vs Legacy)")
    print("="*60)
    
    from src.trading.ppo_agent import PPOTrader
    
    prices = df['Close'].values
    
    # Legacy mode
    print("\n📊 Training LEGACY mode (10 features)...")
    trader_legacy = PPOTrader(
        prices=prices,
        initial_investment=100_000_000,
        use_enhanced_features=False
    )
    trader_legacy.train(episodes=episodes, verbose=False)
    legacy_result = trader_legacy.backtest()
    
    # Enhanced mode
    print("📊 Training ENHANCED mode (14 features)...")
    trader_enhanced = PPOTrader(
        prices=prices,
        initial_investment=100_000_000,
        use_enhanced_features=True,
        ohlcv_df=df
    )
    trader_enhanced.train(episodes=episodes, verbose=False)
    enhanced_result = trader_enhanced.backtest()
    
    # Comparison
    print("\n" + "-"*60)
    print("📊 COMPARISON RESULTS")
    print("-"*60)
    print(f"{'Metric':<20} {'Legacy':>15} {'Enhanced':>15} {'Diff':>12}")
    print("-"*60)
    
    metrics = ['total_return', 'max_drawdown', 'sharpe_ratio', 'win_rate']
    for m in metrics:
        legacy_val = legacy_result['performance'][m]
        enhanced_val = enhanced_result['performance'][m]
        diff = enhanced_val - legacy_val
        sign = "+" if diff > 0 else ""
        
        if m == 'max_drawdown':
            diff = -diff  # Lower drawdown is better
            sign = "+" if diff > 0 else ""
        
        print(f"{m:<20} {legacy_val:>14.2f}% {enhanced_val:>14.2f}% {sign}{diff:>10.2f}%")
    
    return legacy_result, enhanced_result


def test_patchtst_bbca(df, epochs=10):
    """Test PatchTST dengan data BBCA."""
    print("\n" + "="*60)
    print("TEST 4: PatchTST Feature Preparation")
    print("="*60)
    
    from src.models.patchtst_model import prepare_patchtst_input, get_patchtst_feature_info
    
    prices = df['Close'].values
    
    # Get feature info
    info = get_patchtst_feature_info(use_scores=True)
    print(f"\n📋 PatchTST Features ({info['n_features']} dimensions):")
    for name, desc in info['features']:
        print(f"  - {name}: {desc}")
    
    # Prepare features
    print(f"\n🔧 Preparing features (sequence_length=60)...")
    X = prepare_patchtst_input(
        prices=prices,
        ohlcv_df=df,
        use_scores=True,
        sequence_length=60
    )
    
    print(f"✓ Input shape: {X.shape}")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Sequence length: {X.shape[1]}")
    print(f"  - Features: {X.shape[2]}")
    
    # Check stationarity
    log_returns = X[:, :, 0]  # First feature is log returns
    mean_per_sample = np.mean(log_returns, axis=1)
    print(f"\n✓ Log Returns mean across samples: {np.mean(mean_per_sample):.6f} (should be ~0)")
    print(f"✓ No NaN in features: {not np.any(np.isnan(X))}")
    
    return X


if __name__ == "__main__":
    print("="*60)
    print("🏦 BBCA STOCK MODEL TEST")
    print("="*60)
    
    try:
        # Download BBCA data
        df = download_bbca_data(period="2y")
        
        # Test 1: Feature Engineering
        engineer, scores = test_feature_engineering_bbca(df)
        
        # Test 2: PPO with enhanced features
        trader, backtest = test_ppo_bbca(df, episodes=20)
        
        # Test 3: A/B Comparison
        legacy, enhanced = test_ppo_comparison(df, episodes=20)
        
        # Test 4: PatchTST preparation
        X = test_patchtst_bbca(df)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
