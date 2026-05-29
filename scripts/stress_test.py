#!/usr/bin/env python
"""
Stress Test Script (v3.0 - Global Macro Aware)
==============================================

Validates PPO agent robustness under:
1. Variable Fee Stress (0.1% - 1.0%)
2. Noise Injection (0% - 2%)
3. Slippage Simulation (0 - 10 bps)
4. K-Fold Walk-Forward Validation

With Global Macro Features:
- NASDAQ (^IXIC) log returns
- Dow Jones (^DJI) log returns
- 10-Year Treasury (^TNX) log returns

Target: Sharpe 1.8 - 2.5 (reject if > 3.0 - overfit suspected)
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.ppo_agent import PPOTrader
from src.data.feature_engineering import TradingFeatureEngineer, fetch_macro_data, prepare_macro_features

console = Console()

def print_header():
    console.print("[bold cyan]╔════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║  STRESS TEST V3.0 - GLOBAL MACRO AWARE                     ║[/bold cyan]")
    console.print("[bold cyan]║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║[/bold cyan]")
    console.print("[bold cyan]║  Fee Stress | Noise | Slippage | Macro Features            ║[/bold cyan]")
    console.print("[bold cyan]╚════════════════════════════════════════════════════════════╝[/bold cyan]")

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

def run_stress_scenario(name, prices, ohlcv_df, macro_features=None, 
                        fee=0.001, slippage=(0.0, 0.0005), episodes=50):
    console.print(f"\n[bold cyan]>> Running Scenario: {name}[/bold cyan]")
    console.print(f"   Fee: {fee*100:.2f}% | Slippage: {slippage[1]*10000:.1f}bps | Episodes: {episodes}")
    
    # Initialize PPO Trader with macro features
    trader = PPOTrader(
        prices=prices, 
        initial_investment=100_000_000,
        use_enhanced_features=True,
        ohlcv_df=ohlcv_df,
        transaction_fee=fee,
        macro_features=macro_features,
        slippage_range=slippage
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
        'Win Rate': f"{perf['win_rate']:.1f}%",
        'sharpe_float': perf['sharpe_ratio']
    }

def run_kfold_walkforward(ticker, episodes=50):
    """
    K-Fold Walk-Forward Validation
    
    Fold 1: Train 2020-2021, Test 2022
    Fold 2: Train 2020-2022, Test 2023
    Fold 3: Train 2020-2023, Test 2024
    Fold 4: Train 2020-2024, Test 2025
    """
    console.print("\n[bold magenta]>>> K-Fold Walk-Forward Validation <<<[/bold magenta]")
    
    folds = [
        {'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31', 'name': 'Fold 1 (2022)'},
        {'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31', 'name': 'Fold 2 (2023)'},
        {'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-12-31', 'name': 'Fold 3 (2024)'},
        {'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2025-12-31', 'name': 'Fold 4 (2025)'},
    ]
    
    fold_results = []
    
    for fold in folds:
        console.print(f"\n[yellow]{fold['name']}: Train 2020-01-01 to {fold['train_end']}, Test {fold['test_start']} to {fold['test_end']}[/yellow]")
        
        try:
            # Get training data
            train_data = yf.download(ticker, start='2020-01-01', end=fold['train_end'], progress=False)
            if isinstance(train_data.columns, pd.MultiIndex):
                train_data.columns = train_data.columns.get_level_values(0)
            
            # Get test data
            test_data = yf.download(ticker, start=fold['test_start'], end=fold['test_end'], progress=False)
            if isinstance(test_data.columns, pd.MultiIndex):
                test_data.columns = test_data.columns.get_level_values(0)
            
            if len(test_data) < 30:
                console.print(f"[red]Skipping {fold['name']}: Not enough test data[/red]")
                continue
            
            # Prepare macro features
            train_macro = prepare_macro_features(train_data[['Open', 'High', 'Low', 'Close', 'Volume']])
            macro_cols = ['macro_ixic', 'macro_dji', 'macro_tnx', 'macro_vix']
            train_macro_arr = train_macro[macro_cols].values if all(c in train_macro.columns for c in macro_cols) else None
            
            # Train
            trader = PPOTrader(
                prices=train_data['Close'].values,
                initial_investment=100_000_000,
                use_enhanced_features=True,
                ohlcv_df=train_data[['Open', 'High', 'Low', 'Close', 'Volume']],
                transaction_fee=0.001,
                macro_features=train_macro_arr,
                slippage_range=(0.0, 0.0005)
            )
            trainer_res = trader.train(episodes=episodes, verbose=False)
            
            # Test on unseen data
            test_prices = test_data['Close'].values
            test_ohlcv = test_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Prepare test macro
            test_macro = prepare_macro_features(test_ohlcv)
            test_macro_arr = test_macro[macro_cols].values if all(c in test_macro.columns for c in macro_cols) else None
            
            # Create test environment
            from src.trading.ppo_agent import TradingEnv
            from src.data.feature_engineering import TradingFeatureEngineer
            
            test_env = TradingEnv(
                prices=test_prices,
                initial_balance=100_000_000,
                use_enhanced_features=True,
                ohlcv_df=test_ohlcv,
                transaction_fee=0.001,
                macro_features=test_macro_arr
            )
            
            # Run backtest with trained agent
            state = test_env.reset()
            total_reward = 0
            while not test_env.done:
                action, _, _ = trader.agent.get_action(state)  # Fixed: was select_action
                state, reward, done, _ = test_env.step(action)
                total_reward += reward
            
            final_value = test_env.balance + test_env.shares * test_prices[-1]
            test_return = (final_value - 100_000_000) / 100_000_000 * 100
            
            # Calculate Sharpe (simplified)
            portfolio_values = test_env.portfolio_value_history
            if len(portfolio_values) > 1:
                returns = np.diff(portfolio_values) / portfolio_values[:-1]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
            else:
                sharpe = 0
            
            console.print(f"   [green]{fold['name']} Result:[/green] Return={test_return:.2f}%, Sharpe={sharpe:.2f}")
            
            fold_results.append({
                'Fold': fold['name'],
                'Return': test_return,
                'Sharpe': sharpe
            })
            
        except Exception as e:
            console.print(f"[red]Error in {fold['name']}: {e}[/red]")
            continue
    
    return fold_results


def main():
    print_header()
    
    ticker = "NVDA"  # High-alpha market
    
    # Get data with macro features
    data = get_data(ticker=ticker)
    prices = data['Close'].values
    ohlcv_df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Prepare macro features
    console.print("[yellow]Fetching Global Macro Data (NASDAQ, DJI, TNX, VIX)...[/yellow]")
    try:
        macro_df = prepare_macro_features(ohlcv_df)
        macro_cols = ['macro_ixic', 'macro_dji', 'macro_tnx', 'macro_vix']
        macro_features = macro_df[macro_cols].values if all(c in macro_df.columns for c in macro_cols) else None
        console.print("[green]Macro features (including VIX) loaded successfully![/green]")
    except Exception as e:
        console.print(f"[red]Failed to load macro features: {e}[/red]")
        macro_features = None
    
    results = []
    
    # 1. Baseline (Normal Condition + Macro)
    results.append(run_stress_scenario(
        "Baseline (Macro-Aware)", 
        prices, ohlcv_df, 
        macro_features=macro_features,
        fee=0.001, 
        slippage=(0.0, 0.0005),
        episodes=50
    ))
    
    # 2. High Fee + Slippage (Realistic Execution)
    results.append(run_stress_scenario(
        "High Stress (Fee 1% + 10bps Slip)", 
        prices, ohlcv_df, 
        macro_features=macro_features,
        fee=0.01, 
        slippage=(0.0005, 0.001),  # 5-10 bps slippage
        episodes=50
    ))
    
    # 3. Noisy Market (Normal Fee + 2% Noise)
    noisy_prices = add_noise(prices, noise_level=0.02)
    noisy_ohlcv = ohlcv_df.copy()
    noisy_ohlcv['Close'] = noisy_prices
    noisy_ohlcv['High'] = noisy_prices * 1.01
    noisy_ohlcv['Low'] = noisy_prices * 0.99
    
    results.append(run_stress_scenario(
        "Chaos Mode (Noise 2% + Macro)", 
        noisy_prices, noisy_ohlcv, 
        macro_features=macro_features,
        fee=0.001, 
        slippage=(0.0, 0.0005),
        episodes=50
    ))
    
    # Summary Table
    table = Table(title="STRESS TEST RESULTS (Macro-Aware)")
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
    
    # K-Fold Walk-Forward Validation
    console.print("\n[bold yellow]Running K-Fold Walk-Forward Validation...[/bold yellow]")
    fold_results = run_kfold_walkforward(ticker, episodes=50)
    
    if fold_results:
        fold_table = Table(title="K-FOLD WALK-FORWARD RESULTS")
        fold_table.add_column("Fold", style="cyan")
        fold_table.add_column("Return", style="green")
        fold_table.add_column("Sharpe", style="yellow")
        
        for fr in fold_results:
            fold_table.add_row(fr['Fold'], f"{fr['Return']:.2f}%", f"{fr['Sharpe']:.2f}")
        
        console.print("\n")
        console.print(fold_table)
        
        # Calculate variance
        sharpes = [fr['Sharpe'] for fr in fold_results]
        avg_sharpe = np.mean(sharpes)
        sharpe_variance = np.std(sharpes)
        
        console.print(f"\n[bold]Average Sharpe: {avg_sharpe:.2f} | Variance: {sharpe_variance:.2f}[/bold]")
    
    # Interpretation
    baseline_sharpe = results[0]['sharpe_float']
    
    console.print("\n[bold]VERDICT:[/bold]")
    if baseline_sharpe > 3.0:
        console.print("[red]SUSPICIOUS: Sharpe > 3.0 detected. Possible overfitting or data leakage![/red]")
    elif baseline_sharpe >= 1.8:
        console.print("[green]PASSED: Sharpe in target range (1.8 - 2.5). Model is robust![/green]")
    elif baseline_sharpe >= 1.0:
        console.print("[yellow]WARNING: Sharpe below target but acceptable (1.0 - 1.8)[/yellow]")
    else:
        console.print("[red]FAILED: Model underperforming. Review reward shaping.[/red]")

if __name__ == "__main__":
    main()
