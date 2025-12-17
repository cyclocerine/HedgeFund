#!/usr/bin/env python
"""
Test PPO Agent dengan data saham real dari yfinance
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.trading.ppo_agent import PPOTrader

def add_simple_features(close_prices, volume):
    """Tambahkan fitur sederhana untuk PPO"""
    # Create DataFrame from arrays
    df = pd.DataFrame({
        'Close': close_prices,
        'Volume': volume
    })
    
    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Price relative to SMA
    df['Price_SMA_Ratio'] = df['Close'] / df['SMA_20']
    
    # Volatility (20-day rolling std of returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Price momentum (20-day)
    df['Momentum'] = df['Close'].pct_change(periods=20)
    
    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # RSI simple calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def main():
    print('=== PPO Agent dengan Data BBCA.JK ===')
    print()

    # Download data
    print('Downloading BBCA.JK data...')
    ticker = yf.Ticker('BBCA.JK')
    data = ticker.history(start='2023-01-01', end='2024-12-31')
    print(f'Data shape: {data.shape}')
    print(f'Columns: {data.columns.tolist()}')

    # Extract Close and Volume
    close_prices = data['Close'].values
    volume = data['Volume'].values
    
    # Tambahkan fitur sederhana
    feature_df = add_simple_features(close_prices, volume)

    # Pilih fitur untuk state PPO Agent
    feature_columns = ['Daily_Return', 'Price_SMA_Ratio', 'Volatility', 
                       'Momentum', 'Volume_Change', 'RSI']

    # Hapus baris dengan NaN
    feature_df = feature_df.dropna()
    print(f'Data shape after dropna: {feature_df.shape}')

    # Get features as numpy array
    features = feature_df[feature_columns].values
    prices = feature_df['Close'].values
    
    # Replace inf/nan values
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    print(f'Prices shape: {prices.shape}')
    print(f'Features shape: {features.shape}')

    # Setup PPO Trader
    ppo_trader = PPOTrader(
        prices=prices,
        features=features,
        initial_investment=10000000  # 10 juta
    )

    # Train PPO Agent
    print()
    print('Training PPO Agent (30 episodes)...')
    train_results = ppo_trader.train(episodes=30, verbose=True)

    # Backtest
    print()
    print('Running backtest...')
    backtest_results = ppo_trader.backtest()

    # Print hasil
    perf = backtest_results['performance']
    print()
    print('=' * 50)
    print('PPO Agent Results:')
    print('=' * 50)
    print(f"  Initial Investment: Rp {perf['initial_investment']:,.0f}")
    print(f"  Final Value: Rp {perf['final_value']:,.0f}")
    print(f"  Total Return: {perf['total_return']:.2f}%")
    print(f"  Max Drawdown: {perf['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
    print(f"  Win Rate: {perf['win_rate']:.2f}%")
    print(f"  Num Trades: {perf['num_trades']}")
    print('=' * 50)

    # Plot hasil
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Portfolio Value
    plt.subplot(2, 1, 1)
    plt.plot(backtest_results['portfolio_values'], label='Portfolio Value', linewidth=2, color='blue')
    plt.title('PPO Agent - Portfolio Value Over Time (BBCA.JK)')
    plt.xlabel('Day')
    plt.ylabel('Portfolio Value (Rp)')
    plt.grid(True)
    plt.legend()
    
    # Subplot 2: Actions
    plt.subplot(2, 1, 2)
    plot_prices = prices[-len(backtest_results['actions']):]
    plt.plot(plot_prices, label='Price', alpha=0.7, color='gray')
    
    actions = backtest_results['actions']
    buy_idx = [i for i, a in enumerate(actions) if a == 1]
    sell_idx = [i for i, a in enumerate(actions) if a == 2]
    
    if buy_idx:
        plt.scatter(buy_idx, [plot_prices[i] for i in buy_idx], marker='^', color='green', s=100, label='Buy', zorder=5)
    if sell_idx:
        plt.scatter(sell_idx, [plot_prices[i] for i in sell_idx], marker='v', color='red', s=100, label='Sell', zorder=5)
    
    plt.title('PPO Agent - Trading Actions on BBCA.JK')
    plt.xlabel('Day')
    plt.ylabel('Price (Rp)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Simpan plot
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/ppo_bbca_result.png", dpi=300)
    print()
    print('Plot saved to results/ppo_bbca_result.png')
    print()
    print('PPO Agent test completed!')

if __name__ == "__main__":
    main()
