#!/usr/bin/env python
"""
Test PPO Agent
============

Script untuk menguji fungsi PPO Agent.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading.ppo_agent import PPOTrader, TradingEnv

def test_trading_env():
    """Menguji TradingEnv dengan fokus pada perhitungan volatilitas"""
    print("\n=== Menguji TradingEnv ===")
    
    # Generate sample data
    np.random.seed(42)
    days = 100
    
    # Generate random price series dengan trend
    prices = np.cumsum(np.random.normal(0.001, 0.02, days)) + 10
    
    # Buat feature sederhana
    features = np.random.rand(days, 3)  # 3 feature acak
    
    # Inisialisasi environment
    env = TradingEnv(prices=prices, features=features, initial_balance=10000)
    
    # Test reset dan observasi awal
    obs = env.reset()
    print(f"Observasi awal shape: {obs.shape}")
    
    # Test beberapa langkah
    for _ in range(5):
        action = np.random.randint(0, 3)  # 0=hold, 1=buy, 2=sell
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.4f}, Portfolio Value: {info['portfolio_value']:.2f}")
    
    print("TradingEnv berjalan tanpa error!")

def test_ppo_agent_backtest():
    """Menguji PPO Agent dengan fokus pada backtest"""
    print("\n=== Menguji PPO Agent Backtest ===")
    
    # Generate sample data
    np.random.seed(42)
    days = 100
    
    # Generate random price series dengan trend
    prices = np.cumsum(np.random.normal(0.001, 0.02, days)) + 10
    
    # Buat PPO Trader
    ppo_trader = PPOTrader(prices=prices, initial_investment=10000)
    
    # Latih model dengan episode minimal
    print("Melatih PPO agent (episode minimal)...")
    train_results = ppo_trader.train(episodes=2, max_steps=20)
    
    # Backtest
    print("Menjalankan backtest...")
    backtest_results = ppo_trader.backtest()
    
    # Print hasil
    performance = backtest_results['performance']
    print(f"Return: {performance['total_return']:.2f}%")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
    print(f"Jumlah Transaksi: {len(backtest_results['trades'])}")
    
    # Plot hasil
    plt.figure(figsize=(10, 6))
    plt.plot(backtest_results['portfolio_values'], label='Portfolio Value')
    plt.title('PPO Agent Backtest Results')
    plt.xlabel('Day')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    
    # Simpan plot ke file
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/ppo_backtest_result.png", dpi=300)
    print("Plot saved to results/ppo_backtest_result.png")

if __name__ == "__main__":
    print("=== PPO Agent Test ===")
    
    try:
        test_trading_env()
        test_ppo_agent_backtest()
        print("\n✅ All PPO tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during PPO tests: {str(e)}")
        import traceback
        traceback.print_exc() 