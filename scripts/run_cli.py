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

def save_results(predictor, y_true, y_pred, forecast, args, signals=None):
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
            df = pd.DataFrame({
                'Day': range(len(y_true)),
                'Actual': y_true,
                'Predicted': y_pred
            })
            
            # Tambahkan forecast
            forecast_df = pd.DataFrame({
                'Day': range(len(y_true), len(y_true) + len(forecast)),
                'Forecast': forecast
            })
            
            # Jika ada signals, tambahkan ke forecast
            if signals:
                forecast_df['Signal'] = [s.get('action', 'hold') for s in signals]
                forecast_df['Confidence'] = [s.get('confidence', 0) for s in signals]
            
            df.to_csv(f"{filename}_historical.csv", index=False)
            forecast_df.to_csv(f"{filename}_forecast.csv", index=False)
            print_success(f"Hasil disimpan ke {filename}_historical.csv dan {filename}_forecast.csv")
        except Exception as e:
            print_warning(f"Gagal menyimpan hasil ke CSV: {str(e)}")
            
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
        
        # Simpan plot
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values, label='Nilai Portfolio', color='blue')
        plt.title(f'{args.ticker} - Backtest dengan {args.strategy}')
        plt.xlabel('Hari')
        plt.ylabel('Nilai Portfolio')
        plt.grid(True)
        plt.savefig(f"{filename}_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print_success(f"Plot backtest disimpan ke {filename}_plot.png")
        
    except Exception as e:
        print_warning(f"Gagal menyimpan hasil backtest: {str(e)}")

def generate_ppo_signals(prices, forecast, initial_investment=10000000, episodes=30, ohlcv_df=None):
    """Generate trading signals menggunakan PPO agent dengan enhanced features"""
    print_info(f"Melatih PPO agent ({episodes} episodes) untuk menghasilkan sinyal trading...")
    
    # Check if enhanced features available
    try:
        from src.data.feature_engineering import TradingFeatureEngineer
        use_enhanced = True
        print_info("Menggunakan Enhanced Features (MACD, Stoch RSI, BB, Volume scores)")
    except ImportError:
        use_enhanced = False
        print_warning("Enhanced features tidak tersedia, menggunakan legacy mode")
    
    # Setup PPO with enhanced or legacy features
    if use_enhanced and ohlcv_df is not None:
        ppo_trader = PPOTrader(
            prices=prices,
            initial_investment=initial_investment,
            use_enhanced_features=True,
            ohlcv_df=ohlcv_df
        )
    else:
        # Legacy mode - buat features sederhana
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
        # Ambil action terakhir dari training
        if len(actions) > 0:
            recent_actions = actions[-min(20, len(actions)):]
            # Hitung probabilitas action
            buy_count = sum(1 for a in recent_actions if a == 1)
            sell_count = sum(1 for a in recent_actions if a == 2)
            total = len(recent_actions)
            
            buy_prob = buy_count / total
            sell_prob = sell_count / total
            
            if buy_prob > 0.4:
                signal_action = 'buy'
                confidence = 50 + buy_prob * 50
            elif sell_prob > 0.4:
                signal_action = 'sell'
                confidence = 50 + sell_prob * 50
            else:
                signal_action = 'hold'
                confidence = 50
        else:
            signal_action = 'hold'
            confidence = 50
        
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
    parser.add_argument("--ppo-episodes", type=int, default=50, help="Jumlah episode training PPO (default: 50)")
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
                    ohlcv_df=ohlcv_df
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
                save_results(predictor, y_true, y_pred, forecast, args, trading_signals)
                
        elif args.mode == 'backtest':
            # Step 6: Run backtest
            print_step(6, total_steps, f"Menjalankan backtest dengan strategi {args.strategy}...")
            
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
                elif args.strategy == 'PPO':
                    param_ranges = {
                        'actor_lr': [0.0001, 0.0003, 0.001],
                        'critic_lr': [0.0005, 0.001, 0.002],
                        'gamma': [0.95, 0.97, 0.99],
                        'clip_ratio': [0.1, 0.2, 0.3],
                        'episodes': [5, 10]
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