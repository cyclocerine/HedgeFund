import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.ppo_agent import PPOTrader
from src.data.feature_engineering import TradingFeatureEngineer

console = Console()

def print_header():
    console.print("[bold red]╔════════════════════════════════════════════════════════════╗[/bold red]")
    console.print("[bold red]║  ALGORITHMIC TRADING STRESS TEST (BRUTAL MODE)             ║[/bold red]")
    console.print("[bold red]║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║[/bold red]")
    console.print("[bold red]║  Fee Stress | Noise Injection | Reality Check              ║[/bold red]")
    console.print("[bold red]╚════════════════════════════════════════════════════════════╝[/bold red]")

def get_data(ticker="NVDA", start_date="2020-01-01", end_date="2025-01-01"):
    console.print(f"[yellow]Downloading data for {ticker}...[/yellow]")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Ensure standard columns
    if 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']
    
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")
            
    return data

def add_noise(prices, noise_level=0.01):
    """Add Gaussian noise to prices"""
    noise = np.random.normal(0, noise_level, len(prices))
    noisy_prices = prices * (1 + noise)
    return noisy_prices

def run_stress_scenario(name, prices, ohlcv_df, fee=0.001, episodes=50):
    console.print(f"\n[bold cyan]>> Running Scenario: {name}[/bold cyan]")
    console.print(f"   Fee: {fee*100:.2f}% | Episodes: {episodes}")
    
    # Initialize PPO Trader with specific fee
    trader = PPOTrader(
        prices=prices, 
        initial_investment=100_000_000,
        use_enhanced_features=True,
        ohlcv_df=ohlcv_df,
        transaction_fee=fee
    )
    
    # Train
    train_res = trader.train(episodes=episodes, verbose=False)
    avg_reward = np.mean(train_res['episode_rewards'][-10:])
    
    # Backtest
    backtest = trader.backtest()
    perf = backtest['performance']
    
    console.print(f"   [green]Result:[/green] Return={perf['total_return']:.2f}% | Sharpe={perf['sharpe_ratio']:.2f} | MDD={perf['max_drawdown']:.2f}%")
    
    return {
        'Scenario': name,
        'Fee': f"{fee*100:.1f}%",
        'Total Return': f"{perf['total_return']:.2f}%",
        'Sharpe Ratio': f"{perf['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{perf['max_drawdown']:.2f}%",
        'Trades': perf['num_trades'],
        'Win Rate': f"{perf['win_rate']:.1f}%"
    }

def main():
    print_header()
    
    data = get_data()
    prices = data['Close'].values
    ohlcv_df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    results = []
    
    # 1. Baseline (Normal Condition)
    results.append(run_stress_scenario(
        "Baseline (Normal)", 
        prices, ohlcv_df, 
        fee=0.001, 
        episodes=50
    ))
    
    # 2. High Fee (1.0% - Worst Case Slippage)
    results.append(run_stress_scenario(
        "High Stress (Fee 1.0%)", 
        prices, ohlcv_df, 
        fee=0.01, 
        episodes=50
    ))
    
    # 3. Noisy Market (Normal Fee + 2% Noise)
    noisy_prices = add_noise(prices, noise_level=0.02)
    # Reconstruct OHLCV for noise (approx)
    noisy_ohlcv = ohlcv_df.copy()
    noisy_ohlcv['Close'] = noisy_prices
    noisy_ohlcv['High'] = noisy_prices * 1.01
    noisy_ohlcv['Low'] = noisy_prices * 0.99
    
    results.append(run_stress_scenario(
        "Chaos Mode (Noise 2%)", 
        noisy_prices, noisy_ohlcv, 
        fee=0.001, 
        episodes=50
    ))
    
    # Summary Table
    table = Table(title="STRESS TEST RESULTS")
    table.add_column("Scenario", style="cyan", no_wrap=True)
    table.add_column("Fee", style="magenta")
    table.add_column("Return", style="green")
    table.add_column("Sharpe", style="yellow")
    table.add_column("MDD", style="red")
    table.add_column("Win Rate", style="blue")
    
    for res in results:
        table.add_row(
            res['Scenario'], res['Fee'], res['Total Return'], 
            res['Sharpe Ratio'], res['Max Drawdown'], res['Win Rate']
        )
        
    console.print("\n")
    console.print(table)
    
    # Interpretation
    baseline_sharpe = float(results[0]['Sharpe Ratio'])
    stress_sharpe = float(results[1]['Sharpe Ratio'])
    
    console.print("\n[bold]VERDICT:[/bold]")
    if stress_sharpe > 2.0:
        console.print("[green]PASSED: Model is robust against high fees![/green]")
    elif stress_sharpe > 1.0:
        console.print("[yellow]WARNING: Model struggles with high fees but survives.[/yellow]")
    else:
        console.print("[red]FAILED: Model breaks under stress. Overfitting suspected.[/red]")

if __name__ == "__main__":
    main()
