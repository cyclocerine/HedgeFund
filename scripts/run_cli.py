#!/usr/bin/env python
"""
CLI Application
=============

Script untuk menjalankan aplikasi prediksi saham dengan antarmuka command line.
Mendukung PatchTST prediction dan PPO trading signals.
"""

import argparse
from datetime import datetime, timedelta
import sys
import os

# Must be set before any other imports to fully suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

import time
import warnings

# Suppress all warnings for cleaner CLI output
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.layout import Layout
    from rich.align import Align
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("[!] Rich library tidak tersedia. Menggunakan output standar.")

from src.models.predictor import StockPredictor
from src.trading.backtest import Backtester
from src.trading.optimizer import StrategyOptimizer
from src.trading.ppo_agent import PPOTrader

# Inisialisasi console untuk tampilan yang lebih menarik
console = Console() if RICH_AVAILABLE else None

# Spinner styles yang menarik
SPINNERS = ['dots', 'dots2', 'dots3', 'line', 'star', 'growVertical', 'bounce']


def create_progress():
    """Create a rich progress bar with beautiful styling."""
    if not RICH_AVAILABLE:
        return None
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="cyan", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False
    )

def run_with_spinner(description, func, *args, **kwargs):
    """Run a function with an animated spinner."""
    if RICH_AVAILABLE:
        with console.status(f"[bold cyan]{description}...", spinner="dots") as status:
            result = func(*args, **kwargs)
        return result
    else:
        print(f"⏳ {description}...")
        result = func(*args, **kwargs)
        return result

def print_header():
    if RICH_AVAILABLE:
        # Animated header dengan gradient
        header_content = Text()
        header_content.append("╔════════════════════════════════════════════════════════════╗\n", style="bright_blue")
        header_content.append("║", style="bright_blue")
        header_content.append("  APLIKASI PREDIKSI HARGA SAHAM", style="bold cyan")
        header_content.append("                              ║\n", style="bright_blue")
        header_content.append("║", style="bright_blue")
        header_content.append("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", style="dim white")
        header_content.append("  ║\n", style="bright_blue")
        header_content.append("║", style="bright_blue")
        header_content.append("  PatchTST Deep Learning", style="bright_magenta")
        header_content.append(" | ", style="dim")
        header_content.append("PPO Trading", style="bright_yellow")
        header_content.append(" | ", style="dim")
        header_content.append("CUDA", style="bright_green")
        header_content.append("         ║\n", style="bright_blue")
        header_content.append("╚════════════════════════════════════════════════════════════╝", style="bright_blue")
        
        console.print(header_content)
        console.print()
    else:
        print("=" * 60)
        print("  APLIKASI PREDIKSI HARGA SAHAM - MODE CLI")
        print("  Powered by PatchTST + PPO | CUDA Accelerated")
        print("=" * 60)
        print()

def print_step(step_number, total_steps, step_name):
    if RICH_AVAILABLE:
        step_text = Text()
        step_text.append(f"[{step_number}/{total_steps}] ", style="bright_blue")
        step_text.append(">> ", style="bright_yellow")
        step_text.append(step_name, style="bright_white")
        console.print(step_text)
    else:
        print(f"[{step_number}/{total_steps}] >> {step_name}")

def print_success(message):
    if RICH_AVAILABLE:
        console.print(f"[OK] {message}", style="bright_green")
    else:
        print(f"[OK] {message}")

def print_error(message):
    if RICH_AVAILABLE:
        console.print(f"[ERROR] {message}", style="bold red")
    else:
        print(f"[ERROR] {message}")

def print_warning(message):
    if RICH_AVAILABLE:
        console.print(f"[WARN] {message}", style="bright_yellow")
    else:
        print(f"[WARN] {message}")

def print_info(message):
    if RICH_AVAILABLE:
        console.print(f"[INFO] {message}", style="bright_blue")
    else:
        print(f"[INFO] {message}")

def print_model_metrics(metrics):
    """Menampilkan metrik evaluasi model"""
    if RICH_AVAILABLE:
        table = Table.grid(padding=1)
        table.add_column(style="bright_blue")
        table.add_column(style="bright_white")
        
        table.add_row("MSE", f"{metrics.get('mse', metrics.get('MSE', 0)):.4f}")
        table.add_row("RMSE", f"{metrics.get('rmse', metrics.get('RMSE', 0)):.4f}")
        table.add_row("MAE", f"{metrics.get('mae', metrics.get('MAE', 0)):.4f}")
        table.add_row("R² Score", f"{metrics.get('r2', metrics.get('R2', 0)):.4f}")
        
        panel = Panel(
            table,
            title="Metrik Model",
            border_style="bright_blue",
            box=box.ROUNDED
        )
        console.print(panel)
        console.print()
    else:
        print("\nMetrik Evaluasi Model:")
        print("-" * 40)
        print(f"MSE: {metrics.get('mse', metrics.get('MSE', 0)):.4f}")
        print(f"RMSE: {metrics.get('rmse', metrics.get('RMSE', 0)):.4f}")
        print(f"MAE: {metrics.get('mae', metrics.get('MAE', 0)):.4f}")
        print(f"R² Score: {metrics.get('r2', metrics.get('R2', 0)):.4f}")
        print("-" * 40)

def print_forecast(forecast):
    if RICH_AVAILABLE:
        table = Table(title="Prediksi untuk Hari-hari Berikutnya", box=box.ROUNDED, border_style="bright_blue")
        
        table.add_column("Hari", style="cyan", justify="center")
        table.add_column("Harga", style="green", justify="right")
        table.add_column("Tren", style="magenta", justify="center")
        
        for i, price in enumerate(forecast, 1):
            trend = ""
            trend_style = "green"
            if i > 1:
                prev_price = forecast[i-2]
                pct_change = (price - prev_price) / prev_price * 100
                if pct_change > 0:
                    trend = f"[+] +{pct_change:.2f}%"
                    trend_style = "bright_green"
                else:
                    trend = f"[-] {pct_change:.2f}%"
                    trend_style = "bright_red"
            
            table.add_row(
                str(i),
                f"Rp {price:,.2f}",
                Text(trend, style=trend_style)
            )
        
        console.print(table)
    else:
        print("\nPrediksi untuk Hari-hari Berikutnya:")
        print("-" * 40)
        for i, price in enumerate(forecast, 1):
            trend = ""
            if i > 1:
                prev_price = forecast[i-2]
                pct_change = (price - prev_price) / prev_price * 100
                if pct_change > 0:
                    trend = f"([+] +{pct_change:.2f}%)"
                else:
                    trend = f"([-] {pct_change:.2f}%)"
            print(f"Hari {i}: Rp {price:,.2f} {trend}")
        print("-" * 40)

def print_forecast_with_signals(forecast, signals):
    """Menampilkan forecast dengan trading signals"""
    if RICH_AVAILABLE:
        table = Table(title="Prediksi dan Sinyal Trading", box=box.ROUNDED, border_style="bright_blue")
        
        table.add_column("Hari", style="cyan", justify="center")
        table.add_column("Harga", style="green", justify="right")
        table.add_column("Tren", style="magenta", justify="center")
        table.add_column("Sinyal", style="yellow", justify="center")
        table.add_column("Confidence", style="bright_blue", justify="right")
        
        for i, (price, signal) in enumerate(zip(forecast, signals), 1):
            trend = ""
            trend_style = "green"
            if i > 1:
                prev_price = forecast[i-2]
                pct_change = (price - prev_price) / prev_price * 100
                if pct_change > 0:
                    trend = f"[+] +{pct_change:.2f}%"
                    trend_style = "bright_green"
                else:
                    trend = f"[-] {pct_change:.2f}%"
                    trend_style = "bright_red"
            
            # Format sinyal trading
            action = signal.get('action', 'hold')
            signal_text = "[BUY]" if action == 'buy' else "[SELL]" if action == 'sell' else "[HOLD]"
            signal_style = "bright_green" if action == 'buy' else "bright_red" if action == 'sell' else "bright_white"
            
            table.add_row(
                str(i),
                f"Rp {price:,.2f}",
                Text(trend, style=trend_style),
                Text(signal_text, style=signal_style),
                f"{signal.get('confidence', 0):.1f}%"
            )
        
        console.print(table)
    else:
        print("\nPrediksi dan Sinyal Trading:")
        print("-" * 60)
        for i, (price, signal) in enumerate(zip(forecast, signals), 1):
            action = signal.get('action', 'hold')
            signal_text = "[BUY]" if action == 'buy' else "[SELL]" if action == 'sell' else "[HOLD]"
            print(f"Hari {i}: Rp {price:,.2f} | {signal_text} ({signal.get('confidence', 0):.1f}%)")
        print("-" * 60)

def print_trading_summary(signals):
    """Menampilkan ringkasan sinyal trading"""
    total_signals = len(signals)
    buy_signals = sum(1 for s in signals if s.get('action') == 'buy')
    sell_signals = sum(1 for s in signals if s.get('action') == 'sell')
    hold_signals = sum(1 for s in signals if s.get('action') == 'hold')
    
    avg_confidence = sum(s.get('confidence', 0) for s in signals) / total_signals if total_signals > 0 else 0
    
    if RICH_AVAILABLE:
        summary_table = Table.grid(padding=1)
        summary_table.add_column(style="bright_blue")
        summary_table.add_column(style="bright_white")
        
        summary_table.add_row("Total Hari", str(total_signals))
        summary_table.add_row("Sinyal Beli", f"{buy_signals} ({buy_signals/total_signals*100:.1f}%)" if total_signals > 0 else "0")
        summary_table.add_row("Sinyal Jual", f"{sell_signals} ({sell_signals/total_signals*100:.1f}%)" if total_signals > 0 else "0")
        summary_table.add_row("Sinyal Tahan", f"{hold_signals} ({hold_signals/total_signals*100:.1f}%)" if total_signals > 0 else "0")
        summary_table.add_row("Rata-rata Confidence", f"{avg_confidence:.1f}%")
        
        summary_panel = Panel(
            summary_table,
            title="Ringkasan Sinyal Trading",
            border_style="bright_blue",
            box=box.ROUNDED
        )
        console.print(summary_panel)
    else:
        print("\nRingkasan Sinyal Trading:")
        print("-" * 40)
        print(f"Total Hari: {total_signals}")
        if total_signals > 0:
            print(f"Sinyal Beli: {buy_signals} ({buy_signals/total_signals*100:.1f}%)")
            print(f"Sinyal Jual: {sell_signals} ({sell_signals/total_signals*100:.1f}%)")
            print(f"Sinyal Tahan: {hold_signals} ({hold_signals/total_signals*100:.1f}%)")
        print(f"Rata-rata Confidence: {avg_confidence:.1f}%")
        print("-" * 40)

def print_backtest_results(results):
    """Mencetak hasil backtest ke console"""
    portfolio_values, trades, performance = results
    
    if RICH_AVAILABLE:
        # Panel untuk ringkasan kinerja
        summary_table = Table(box=None)
        summary_table.add_column("Metrik", style="cyan")
        summary_table.add_column("Nilai", justify="right", style="green")
        
        summary_table.add_row("Investasi Awal", f"Rp {performance['initial_investment']:,.2f}")
        summary_table.add_row("Nilai Akhir", f"Rp {performance['final_value']:,.2f}")
        summary_table.add_row("Return Total", f"{performance['total_return']:.2f}%")
        summary_table.add_row("Maximum Drawdown", f"{performance['max_drawdown']:.2f}%")
        summary_table.add_row("Sharpe Ratio", f"{performance['sharpe_ratio']:.4f}")
        summary_table.add_row("Win Rate", f"{performance['win_rate']:.2f}%")
        summary_table.add_row("Jumlah Transaksi", str(performance['num_trades']))
        
        summary_panel = Panel(
            summary_table,
            title="📈 Ringkasan Hasil Backtest",
            border_style="bright_blue",
            box=box.ROUNDED
        )
        console.print(summary_panel)
    else:
        print("\n📈 Ringkasan Hasil Backtest:")
        print("-" * 40)
        print(f"Investasi Awal: Rp {performance['initial_investment']:,.2f}")
        print(f"Nilai Akhir: Rp {performance['final_value']:,.2f}")
        print(f"Return Total: {performance['total_return']:.2f}%")
        print(f"Maximum Drawdown: {performance['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
        print(f"Win Rate: {performance['win_rate']:.2f}%")
        print(f"Jumlah Transaksi: {performance['num_trades']}")
        print("-" * 40)

def save_results(predictor, y_true, y_pred, forecast, args, signals=None, metrics=None, ppo_results=None):
    """Menyimpan hasil prediksi ke file"""
    try:
        # Buat direktori results jika belum ada
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # Generate nama file berdasarkan ticker dan tanggal
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{args.ticker}_{today}"
        
        # Simpan plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Aktual', color='blue')
        plt.plot(y_pred, label='Prediksi', color='red', linestyle='--')
        plt.plot(range(len(y_true), len(y_true) + len(forecast)), 
                forecast, label='Forecast', color='green', linestyle='-.')
        plt.title(f'{args.ticker} - Prediksi Harga Saham dengan {args.model.upper()}')
        plt.xlabel('Hari')
        plt.ylabel('Harga')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{filename}_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print_success(f"Plot disimpan ke {filename}_plot.png")
        
        # Simpan hasil dalam CSV
        try:
            import pandas as pd
            import json
            
            # 1. Simpan Historical + Prediction
            df = pd.DataFrame({
                'Day': range(len(y_true)),
                'Actual': y_true,
                'Predicted': y_pred
            })
            df.to_csv(f"{filename}_historical.csv", index=False)
            
            # 2. Simpan Forecast
            forecast_df = pd.DataFrame({
                'Day': range(len(y_true), len(y_true) + len(forecast)),
                'Forecast': forecast
            })
            
            # Jika ada signals, tambahkan ke forecast
            if signals:
                forecast_df['Signal'] = [s.get('action', 'hold') for s in signals]
                forecast_df['Confidence'] = [s.get('confidence', 0) for s in signals]
                
            forecast_df.to_csv(f"{filename}_forecast.csv", index=False)
            
            # 3. Simpan Evaluasi Metrics
            if metrics:
                # Convert metrics to native python types for JSON serialization
                serializable_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
                with open(f"{filename}_metrics.json", 'w') as f:
                    json.dump(serializable_metrics, f, indent=4)
            
            # 4. Simpan PPO Results (jika ada)
            if ppo_results:
                # Extract clean performance dict
                ppo_perf = ppo_results.get('performance', {})
                serializable_ppo = {k: float(v) if hasattr(v, 'item') else v for k, v in ppo_perf.items()}
                with open(f"{filename}_ppo_results.json", 'w') as f:
                    json.dump(serializable_ppo, f, indent=4)
                    
            print_success(f"Hasil lengkap disimpan di:")
            print_success(f" - CSV: {filename}_historical.csv & {filename}_forecast.csv")
            if metrics: print_success(f" - Metrics: {filename}_metrics.json")
            if ppo_results: print_success(f" - PPO Results: {filename}_ppo_results.json")
            
        except Exception as e:
            print_warning(f"Gagal menyimpan hasil ke CSV/JSON: {str(e)}")
            
    except Exception as e:
        print_warning(f"Gagal menyimpan hasil: {str(e)}")

def save_backtest_results(predictor, backtest_results, args):
    """Simpan hasil backtest ke file"""
    try:
        # Buat direktori results jika belum ada
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # Generate nama file
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{args.ticker}_{args.strategy}_{today}_backtest"
        
        portfolio_values, trades, performance = backtest_results
        
        # 1. Simpan plot
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values, label='Nilai Portfolio', color='blue')
        plt.title(f'{args.ticker} - Backtest dengan {args.strategy}')
        plt.xlabel('Hari')
        plt.ylabel('Nilai Portfolio')
        plt.grid(True)
        plt.savefig(f"{filename}_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Simpan Metrics ke JSON
        import json
        with open(f"{filename}_metrics.json", 'w') as f:
            json.dump(performance, f, indent=4)
            
        # 3. Simpan Portfolio History ke CSV
        import pandas as pd
        portfolio_df = pd.DataFrame({
            'Day': range(len(portfolio_values)),
            'Portfolio_Value': portfolio_values
        })
        portfolio_df.to_csv(f"{filename}_portfolio.csv", index=False)
        
        # 4. Simpan Trade Log ke CSV
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(f"{filename}_trades.csv", index=False)
        
        print_success(f"Hasil Backtest lengkap disimpan di:")
        print_success(f" - Plot: {filename}_plot.png")
        print_success(f" - Metrics: {filename}_metrics.json")
        print_success(f" - Portfolio: {filename}_portfolio.csv")
        print_success(f" - Trades: {filename}_trades.csv" if trades else " - Trades: Tidak ada trades")
        
    except Exception as e:
        print_warning(f"Gagal menyimpan hasil backtest: {str(e)}")

def run_ppo_backtest(prices, initial_investment, episodes=200, ohlcv_df=None, verbose=True, tune=False, train_noise_level=0.0):
    """
    Run PPO backtest with enhanced features - consistent with predict mode.
    
    Parameters:
    -----------
    prices : np.ndarray
        Array of historical prices
    initial_investment : float
        Initial investment amount
    episodes : int
        Number of training episodes (default: 200)
    ohlcv_df : pd.DataFrame
        OHLCV DataFrame with Open, High, Low, Close, Volume columns
    verbose : bool
        Print training progress
    tune : bool
        Enable hyperparameter tuning for PPO agent
    train_noise_level : float
        Noise injection level for PPO training (0.0-0.1)
        
    Returns:
    --------
    tuple: (portfolio_values, trades, performance)
    """
    from src.trading.ppo_agent import PPOTrader
    
    # Check if enhanced features available
    try:
        from src.data.feature_engineering import TradingFeatureEngineer
        use_enhanced = ohlcv_df is not None
        if use_enhanced:
            print_info("Menggunakan Enhanced Features (MACD, Stoch RSI, BB, Volume scores)")
    except ImportError:
        use_enhanced = False
        print_warning("Enhanced features tidak tersedia, menggunakan legacy mode")

    # Prepare macro features
    macro_features = None
    if use_enhanced and ohlcv_df is not None:
        macro_cols = ['macro_ixic', 'macro_dji', 'macro_tnx', 'macro_vix']
        if all(col in ohlcv_df.columns for col in macro_cols):
            macro_features = ohlcv_df[macro_cols].values
            print_info("Mengintegrasikan 4 Macro Features (VIX, NASDAQ, DJI, TNX)")
    
    # PPO Hyperparameter tuning
    if tune:
        print_info("Melakukan PPO Hyperparameter Tuning...")
        
        # Hyperparameter search space (simplified for efficiency)
        hp_configs = [
            {'lr': 0.0003, 'gamma': 0.99, 'clip_ratio': 0.2},  # Default
            {'lr': 0.0001, 'gamma': 0.99, 'clip_ratio': 0.1},  # Conservative
            {'lr': 0.001, 'gamma': 0.95, 'clip_ratio': 0.3},   # Aggressive
        ]
        
        best_reward = float('-inf')
        best_config = hp_configs[0]
        best_trader = None
        
        tune_episodes = min(50, episodes // 4)  # Quick tune with fewer episodes
        
        for i, config in enumerate(hp_configs):
            print_info(f"Tuning config {i+1}/{len(hp_configs)}: lr={config['lr']}, gamma={config['gamma']}, clip={config['clip_ratio']}")
            
            trader = PPOTrader(
                prices=prices,
                initial_investment=initial_investment,
                use_enhanced_features=use_enhanced,
                ohlcv_df=ohlcv_df,
                train_noise_level=train_noise_level,
                macro_features=macro_features
            )
            
            # Apply hyperparameters - use correct attribute names
            import torch.optim as optim
            trader.agent.optimizer = optim.Adam(trader.agent.network.parameters(), lr=config['lr'])
            trader.agent.gamma = config['gamma']
            trader.agent.clip_ratio = config['clip_ratio']
            
            # Quick train
            results = trader.train(episodes=tune_episodes, verbose=False)
            
            if results['best_reward'] > best_reward:
                best_reward = results['best_reward']
                best_config = config
                best_trader = trader
                print_info(f"  >> New best config! Reward: {best_reward:.4f}")
        
        print_success(f"Best config: lr={best_config['lr']}, gamma={best_config['gamma']}, clip_ratio={best_config['clip_ratio']}")
        
        # Full training with best config
        print_info(f"Training dengan config terbaik ({episodes} episodes)...")
        trader = PPOTrader(
            prices=prices,
            initial_investment=initial_investment,
            use_enhanced_features=use_enhanced,
            ohlcv_df=ohlcv_df,
            train_noise_level=train_noise_level,
            macro_features=macro_features
        )
        
        # Apply best hyperparameters
        trader.agent.optimizer = optim.Adam(trader.agent.network.parameters(), lr=best_config['lr'])
        trader.agent.gamma = best_config['gamma']
        trader.agent.clip_ratio = best_config['clip_ratio']
        
        train_results = trader.train(episodes=episodes, verbose=verbose)
    else:
        # Create PPOTrader with default hyperparameters
        trader = PPOTrader(
            prices=prices,
            initial_investment=initial_investment,
            use_enhanced_features=use_enhanced,
            ohlcv_df=ohlcv_df,
            train_noise_level=train_noise_level,
            macro_features=macro_features
        )
        
        # Train agent
        print_info(f"Melatih PPO agent ({episodes} episodes)...")
        train_results = trader.train(episodes=episodes, verbose=verbose)
    
    # Run backtest
    print_info("Menjalankan backtest PPO...")
    backtest_results = trader.backtest()
    
    # Extract metrics for compatibility
    portfolio_values = backtest_results.get('portfolio_values', [])
    trades = backtest_results.get('trades', [])
    final_value = portfolio_values[-1] if portfolio_values else initial_investment
    
    # Calculate performance metrics
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Max drawdown
        peak = portfolio_values[0]
        max_dd = 0
        for val in portfolio_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            max_dd = max(max_dd, dd)
        
        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Win rate
        buy_prices = []
        sell_values = []
        wins = 0
        losses = 0
        for trade in trades:
            if trade.get('type') == 'BUY':
                buy_prices.append(trade.get('price', 0))
            elif trade.get('type') == 'SELL' and buy_prices:
                sell_price = trade.get('price', 0)
                buy_price = buy_prices.pop(0)
                if sell_price > buy_price:
                    wins += 1
                else:
                    losses += 1
        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    else:
        total_return = 0
        max_dd = 0
        sharpe = 0
        win_rate = 0
    
    performance = {
        'initial_investment': initial_investment,
        'final_value': final_value,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'best_reward': train_results.get('best_reward', 0)
    }
    
    return portfolio_values, trades, performance


def generate_ppo_signals(prices, forecast, initial_investment=10000000, episodes=30, ohlcv_df=None, train_noise_level=0.0):
    """Generate trading signals menggunakan PPO agent dengan enhanced features"""
    print_info(f"Melatih PPO agent ({episodes} episodes) untuk menghasilkan sinyal trading...")
    
    # Check if enhanced features available
    try:
        from src.data.feature_engineering import TradingFeatureEngineer
        use_enhanced = ohlcv_df is not None
        if use_enhanced:
            print_info("Menggunakan Enhanced Features (MACD, Stoch RSI, BB, Volume scores)")
            if train_noise_level > 0:
                print_info(f"Noise Injection: {train_noise_level*100:.1f}%")
    except ImportError:
        use_enhanced = False
        print_warning("Enhanced features tidak tersedia, menggunakan legacy mode")
                
    # Prepare macro features
    macro_features = None
    if use_enhanced and ohlcv_df is not None:
        macro_cols = ['macro_ixic', 'macro_dji', 'macro_tnx', 'macro_vix']
        if all(col in ohlcv_df.columns for col in macro_cols):
            macro_features = ohlcv_df[macro_cols].values
            print_info("Mengintegrasikan 4 Macro Features (VIX, NASDAQ, DJI, TNX)")
    
    # Create PPOTrader based on available features
    if use_enhanced and ohlcv_df is not None:
        ppo_trader = PPOTrader(
            prices=prices,
            initial_investment=initial_investment,
            use_enhanced_features=True,
            ohlcv_df=ohlcv_df,
            train_noise_level=train_noise_level,
            macro_features=macro_features
        )
    else:
        # Prepare legacy features
        import pandas as pd
        df = pd.DataFrame({'Close': prices})
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['Price_SMA_Ratio'] = df['Close'] / df['SMA_20'].replace(0, 1)
        df['Volatility'] = df['Daily_Return'].rolling(window=20, min_periods=1).std().fillna(0.02)
        df['Momentum'] = df['Close'].pct_change(periods=20).fillna(0)
        
        feature_columns = ['Daily_Return', 'Price_SMA_Ratio', 'Volatility', 'Momentum']
        features = df[feature_columns].values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        ppo_trader = PPOTrader(
            prices=prices,
            features=features,
            initial_investment=initial_investment
        )
    
    # Train PPO agent
    ppo_trader.train(episodes=episodes, verbose=True)
    print_success(f"PPO agent selesai dilatih ({episodes} episodes)")
    
    # Backtest untuk mendapatkan performance
    backtest_results = ppo_trader.backtest()
    perf = backtest_results['performance']
    
    print_info(f"PPO Backtest Result: Return={perf['total_return']:.2f}%, Sharpe={perf['sharpe_ratio']:.4f}")
    
    # Generate signals untuk forecast period
    signals = []
    actions = backtest_results['actions']
    
    for i in range(len(forecast)):
        # Generate signal using TRAINED MODEL's softmax output
        import torch
        
        # Get last observation from trained environment
        state = ppo_trader.env._get_observation()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ppo_trader.agent.device)
        
        # Get action probabilities from trained network
        with torch.no_grad():
            action_probs, _ = ppo_trader.agent.network(state_tensor)
            probs = action_probs.cpu().numpy().flatten()
        
        # probs[0] = HOLD, probs[1] = BUY, probs[2] = SELL
        hold_prob = probs[0] if len(probs) > 0 else 0.33
        buy_prob = probs[1] if len(probs) > 1 else 0.33
        sell_prob = probs[2] if len(probs) > 2 else 0.33
        
        # Determine signal based on highest probability
        if buy_prob > 0.5:
            signal_action = 'buy'
            confidence = buy_prob * 100
        elif sell_prob > 0.5:
            signal_action = 'sell'
            confidence = sell_prob * 100
        elif buy_prob > sell_prob and buy_prob > hold_prob:
            signal_action = 'buy'
            confidence = buy_prob * 100
        elif sell_prob > buy_prob and sell_prob > hold_prob:
            signal_action = 'sell'
            confidence = sell_prob * 100
        else:
            signal_action = 'hold'
            confidence = hold_prob * 100
        
        signals.append({
            'action': signal_action,
            'confidence': min(100, max(0, confidence))
        })
    
    return signals, backtest_results

def parse_args():
    parser = argparse.ArgumentParser(description="Aplikasi prediksi harga saham dengan PPO trading signals")
    
    # Parameter wajib
    parser.add_argument("--ticker", required=True, help="Kode saham (contoh: BMRI.JK)")
    parser.add_argument("--mode", required=True, choices=["predict", "backtest"], help="Mode operasi: predict atau backtest")
    
    # Parameter opsional
    parser.add_argument("--model", default="patchtst", choices=["patchtst", "improved_patchtst", "ensemble"], help="Model prediksi: patchtst, improved_patchtst, atau ensemble")
    parser.add_argument("--ensemble", action="store_true", help="Gunakan ensemble model (PatchTST + BiLSTM + XGBoost)")
    parser.add_argument("--start-date", help="Tanggal awal data (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Tanggal akhir data (YYYY-MM-DD)")
    parser.add_argument("--tune", action="store_true", help="Aktifkan hyperparameter tuning")
    parser.add_argument("--ppo", action="store_true", help="Aktifkan PPO trading signals")
    parser.add_argument("--ppo-episodes", type=int, default=200, help="Jumlah episode training PPO (default: 200)")
    parser.add_argument("--train-noise", type=float, default=0.0, help="Noise injection level for PPO training (0.0-0.1, default: 0.0)")
    parser.add_argument("--save-results", action="store_true", help="Simpan hasil prediksi")
    
    # Parameter lookback dan forecast
    parser.add_argument("--lookback", type=int, default=60, help="Jumlah hari historis")
    parser.add_argument("--forecast-days", type=int, default=30, help="Jumlah hari prediksi")
    
    # Parameter khusus backtest
    parser.add_argument("--strategy", default="PPO", choices=["Trend Following", "Mean Reversion", "Predictive", "PPO"], help="Strategi trading")
    parser.add_argument("--optimize", action="store_true", help="Aktifkan optimasi parameter strategi")
    parser.add_argument("--initial-balance", type=float, default=100000000, help="Modal awal untuk backtest")
    
    args = parser.parse_args()
    
    # Set tanggal default jika tidak diisi (6 tahun dari sekarang)
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=6*365)).strftime("%Y-%m-%d")
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    return args

def main():
    print_header()
    
    args = parse_args()
    
    # Print konfigurasi
    print_info(f"Ticker: {args.ticker}")
    print_info(f"Mode: {args.mode.upper()}")
    print_info(f"Periode: {args.start_date} hingga {args.end_date}")
    print_info(f"Model: {args.model.upper()}")
    print_info(f"Lookback: {args.lookback} hari, Forecast: {args.forecast_days} hari")
    if args.ppo:
        print_info("PPO Trading Signals: Aktif")
    if args.mode == 'backtest':
        print_info(f"Strategy: {args.strategy}")
        print_info(f"Initial Balance: Rp {args.initial_balance:,.0f}")
    print()
    
    try:
        total_steps = 7 if args.mode == 'backtest' else (6 if args.ppo else 5)
        
        if RICH_AVAILABLE:
            # Create main progress display
            with Progress(
                SpinnerColumn("dots"),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False
            ) as progress:
                
                # Step 1: Initialize predictor
                task1 = progress.add_task("[cyan]Mengunduh data...", total=100)
                # Determine model type
                model_type = args.model
                use_ensemble = args.ensemble or args.model == 'ensemble'
                if use_ensemble:
                    model_type = 'patchtst'  # Base model for ensemble
                
                predictor = StockPredictor(
                    ticker=args.ticker,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    lookback=args.lookback,
                    forecast_days=args.forecast_days,
                    model_type=model_type,
                    tune_hyperparameters=args.tune,
                    use_ensemble=use_ensemble
                )
                progress.update(task1, completed=50, description="[cyan]Mempersiapkan data...")
                
                # Step 2: Prepare data  
                if not predictor.prepare_data():
                    print_error("Gagal mempersiapkan data")
                    return 1
                progress.update(task1, completed=100, description="[green][OK] Data siap!")
                
                # Step 3: Train model
                model_name = "ENSEMBLE" if (args.ensemble or args.model == 'ensemble') else args.model.upper()
                task2 = progress.add_task(f"[magenta]Melatih {model_name}...", total=100)
                start_time = time.time()
                
                # Simulate training progress with animation
                for i in range(0, 100, 10):
                    time.sleep(0.05)  # Small delay for animation
                    progress.update(task2, completed=i)
                
                predictor.train_model()
                training_time = time.time() - start_time
                progress.update(task2, completed=100, description=f"[green][OK] Model terlatih ({training_time:.1f}s)")
                
                # Step 4: Make predictions
                task3 = progress.add_task("[yellow]Membuat prediksi...", total=100)
                for i in range(0, 50, 10):
                    time.sleep(0.02)
                    progress.update(task3, completed=i)
                    
                y_true, y_pred, forecast = predictor.predict()
                progress.update(task3, completed=100, description="[green][OK] Prediksi selesai!")
                
                # Step 5: Evaluate model
                task4 = progress.add_task("[blue]Mengevaluasi model...", total=100)
                metrics = predictor.evaluate(y_true, y_pred)
                progress.update(task4, completed=100, description="[green][OK] Evaluasi selesai!")
            
            console.print()
        else:
            # Fallback tanpa rich
            print_step(1, total_steps, "Memulai prediktor dan mengunduh data...")
            
            # Determine model type
            model_type = args.model
            use_ensemble = args.ensemble or args.model == 'ensemble'
            if use_ensemble:
                model_type = 'patchtst'  # Base model for ensemble
            
            predictor = StockPredictor(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                lookback=args.lookback,
                forecast_days=args.forecast_days,
                model_type=model_type,
                tune_hyperparameters=args.tune,
                use_ensemble=use_ensemble
            )
            
            print_step(2, total_steps, "Mempersiapkan data...")
            if not predictor.prepare_data():
                print_error("Gagal mempersiapkan data")
                return 1
            print_success("Data berhasil dipersiapkan")
            
            model_name = "ENSEMBLE" if use_ensemble else args.model.upper()
            print_step(3, total_steps, f"Melatih model {model_name}...")
            start_time = time.time()
            predictor.train_model()
            training_time = time.time() - start_time
            print_success(f"Model berhasil dilatih dalam {training_time:.2f} detik")
            
            print_step(4, total_steps, "Membuat prediksi...")
            y_true, y_pred, forecast = predictor.predict()
            print_success("Prediksi berhasil dibuat")
            
            print_step(5, total_steps, "Mengevaluasi model...")
            metrics = predictor.evaluate(y_true, y_pred)
            print_success("Evaluasi selesai")
        
        # Print metrics
        print_model_metrics(metrics)
        
        if args.mode == 'predict':
            # Generate PPO signals if enabled
            trading_signals = None
            ppo_backtest = None
            if args.ppo:
                print_step(6, total_steps, f"Menghasilkan sinyal trading dengan PPO ({args.ppo_episodes} episodes)...")
                
                # Get OHLCV data from predictor if available
                ohlcv_df = predictor.data if hasattr(predictor, 'data') else None
                
                trading_signals, ppo_backtest = generate_ppo_signals(
                    y_true, forecast, 
                    initial_investment=int(args.initial_balance),
                    episodes=args.ppo_episodes,
                    ohlcv_df=ohlcv_df,
                    train_noise_level=args.train_noise
                )
                
                # Print PPO backtest results
                if ppo_backtest:
                    print_backtest_results((
                        ppo_backtest['portfolio_values'],
                        ppo_backtest['trades'],
                        ppo_backtest['performance']
                    ))
                
                print_forecast_with_signals(forecast, trading_signals)
                print_trading_summary(trading_signals)
            else:
                print_forecast(forecast)
            
            # Save results if requested
            if args.save_results:
                save_results(predictor, y_true, y_pred, forecast, args, trading_signals, metrics, ppo_backtest)
                
        elif args.mode == 'backtest':
            # Step 6: Run backtest
            print_step(6, total_steps, f"Menjalankan backtest dengan strategi {args.strategy}...")
            
            # Get OHLCV data from predictor for enhanced features
            ohlcv_df = predictor.data if hasattr(predictor, 'data') else None
            
            # Special handling for PPO strategy - use enhanced features
            if args.strategy == 'PPO':
                # Use the new run_ppo_backtest function with enhanced features
                episodes = args.ppo_episodes if hasattr(args, 'ppo_episodes') else 200
                backtest_results = run_ppo_backtest(
                    prices=y_true,
                    initial_investment=args.initial_balance,
                    episodes=episodes,
                    ohlcv_df=ohlcv_df,
                    verbose=True,
                    tune=args.tune if hasattr(args, 'tune') else False,
                    train_noise_level=args.train_noise if hasattr(args, 'train_noise') else 0.0
                )
            else:
                # Use traditional backtester for other strategies
                backtester = Backtester(y_true, y_pred)
                
                # Optimize strategy parameters if requested
                if args.optimize:
                    print_info("Mengoptimalkan parameter strategi...")
                    optimizer = StrategyOptimizer(y_true, y_pred)
                    
                    # Set parameter ranges berdasarkan strategi
                    if args.strategy == 'Trend Following':
                        param_ranges = {'threshold': [0.005, 0.01, 0.02, 0.03, 0.05]}
                    elif args.strategy == 'Mean Reversion':
                        param_ranges = {
                            'window': [3, 5, 10, 15, 20],
                            'buy_threshold': [0.97, 0.98, 0.99],
                            'sell_threshold': [1.01, 1.02, 1.03]
                        }
                    elif args.strategy == 'Predictive':
                        param_ranges = {
                            'buy_threshold': [1.005, 1.01, 1.02],
                            'sell_threshold': [0.98, 0.99, 0.995]
                        }
                    else:
                        param_ranges = {}
                    
                    best_params, best_performance, best_portfolio, best_trades = optimizer.optimize(
                        args.strategy, param_ranges
                    )
                    
                    print_info(f"Parameter optimal: {best_params}")
                    backtest_results = (best_portfolio, best_trades, best_performance)
                else:
                    # Gunakan parameter default
                    backtest_results = backtester.run(args.strategy)
            
            print_success("Backtest selesai")
            
            # Print backtest results
            print_backtest_results(backtest_results)
            
            # Step 7: Save backtest results if requested
            if args.save_results:
                print_step(7, total_steps, "Menyimpan hasil backtest...")
                save_backtest_results(predictor, backtest_results, args)
        
        # Completion message
        if RICH_AVAILABLE:
            completion = Text()
            completion.append("\n")
            completion.append("╭──────────────────────────────────────╮\n", style="bright_green")
            completion.append("│", style="bright_green")
            completion.append("  ** ", style="bright_yellow")
            completion.append("SELESAI", style="bold bright_green")
            completion.append(" **", style="bright_yellow")
            completion.append("                       │\n", style="bright_green")
            completion.append("│", style="bright_green")
            completion.append("  Terima kasih telah menggunakan!", style="dim white")
            completion.append("    │\n", style="bright_green")
            completion.append("╰──────────────────────────────────────╯", style="bright_green")
            console.print(completion)
        else:
            print("\n** Selesai! **")
        return 0
        
    except Exception as e:
        print_error(f"Terjadi kesalahan: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())