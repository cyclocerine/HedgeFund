#!/usr/bin/env python
"""
CLI Application
=============

Script untuk menjalankan aplikasi prediksi saham dengan antarmuka command line.
"""

import argparse
from datetime import datetime, timedelta
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.syntax import Syntax
from rich import box

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predictor import StockPredictor
from src.trading.backtest import Backtester
from src.trading.optimizer import StrategyOptimizer

# Inisialisasi console untuk tampilan yang lebih menarik
console = Console()

def print_header():
    title = Text("🚀 APLIKASI PREDIKSI HARGA SAHAM", style="bold cyan")
    subtitle = Text("cyclocerine", style="italic blue")
    
    header_panel = Panel(
        Text.assemble(title, "\n", subtitle),
        box=box.DOUBLE,
        border_style="bright_blue",
        padding=(1, 2)
    )
    console.print(header_panel)
    console.print()

def print_step(step_number, total_steps, step_name):
    step_text = Text()
    step_text.append(f"[{step_number}/{total_steps}] ", style="bright_blue")
    step_text.append("🔄 ", style="bright_yellow")
    step_text.append(step_name, style="bright_white")
    console.print(step_text)

def print_success(message):
    console.print(f"✅ {message}", style="bright_green")

def print_error(message):
    console.print(f"❌ ERROR: {message}", style="bold red")

def print_warning(message):
    console.print(f"⚠️ {message}", style="bright_yellow")

def print_info(message):
    console.print(f"ℹ️ {message}", style="bright_blue")

def print_model_metrics(metrics):
    """Menampilkan metrik evaluasi model"""
    table = Table.grid(padding=1)
    table.add_column(style="bright_blue")
    table.add_column(style="bright_white")
    
    table.add_row("MSE", f"{metrics['MSE']:.4f}")
    table.add_row("RMSE", f"{metrics['RMSE']:.4f}")
    table.add_row("MAE", f"{metrics['MAE']:.4f}")
    table.add_row("R2 Score", f"{metrics['R2']:.4f}")
    
    panel = Panel(
        table,
        title="📊 Metrik Model",
        border_style="bright_blue",
        box=box.ROUNDED
    )
    console.print(panel)
    console.print()

def print_forecast(forecast):
    table = Table(title="🔮 Prediksi untuk Hari-hari Berikutnya", box=box.ROUNDED, border_style="bright_blue")
    
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
                trend = f"↗️ +{pct_change:.2f}%"
                trend_style = "bright_green"
            else:
                trend = f"↘️ {pct_change:.2f}%"
                trend_style = "bright_red"
        
        table.add_row(
            str(i),
            f"Rp {price:,.2f}",
            Text(trend, style=trend_style)
        )
    
    console.print(table)

def print_forecast_with_signals(forecast, signals):
    table = Table(title="🔮 Prediksi dan Sinyal Trading", box=box.ROUNDED, border_style="bright_blue")
    
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
                trend = f"↗️ +{pct_change:.2f}%"
                trend_style = "bright_green"
            else:
                trend = f"↘️ {pct_change:.2f}%"
                trend_style = "bright_red"
        
        # Format sinyal trading
        signal_text = "🟢 BELI" if signal['action'] == 'buy' else "🔴 JUAL" if signal['action'] == 'sell' else "⚪ TAHAN"
        signal_style = "bright_green" if signal['action'] == 'buy' else "bright_red" if signal['action'] == 'sell' else "bright_white"
        
        table.add_row(
            str(i),
            f"Rp {price:,.2f}",
            Text(trend, style=trend_style),
            Text(signal_text, style=signal_style),
            f"{signal['confidence']:.1f}%"
        )
    
    console.print(table)

def print_trading_summary(signals):
    # Hitung statistik sinyal
    total_signals = len(signals)
    buy_signals = sum(1 for s in signals if s['action'] == 'buy')
    sell_signals = sum(1 for s in signals if s['action'] == 'sell')
    hold_signals = sum(1 for s in signals if s['action'] == 'hold')
    
    avg_confidence = sum(s['confidence'] for s in signals) / total_signals
    
    # Buat tabel ringkasan
    summary_table = Table.grid(padding=1)
    summary_table.add_column(style="bright_blue")
    summary_table.add_column(style="bright_white")
    
    summary_table.add_row("Total Hari", str(total_signals))
    summary_table.add_row("Sinyal Beli", f"{buy_signals} ({buy_signals/total_signals*100:.1f}%)")
    summary_table.add_row("Sinyal Jual", f"{sell_signals} ({sell_signals/total_signals*100:.1f}%)")
    summary_table.add_row("Sinyal Tahan", f"{hold_signals} ({hold_signals/total_signals*100:.1f}%)")
    summary_table.add_row("Rata-rata Confidence", f"{avg_confidence:.1f}%")
    
    summary_panel = Panel(
        summary_table,
        title="📊 Ringkasan Sinyal Trading",
        border_style="bright_blue",
        box=box.ROUNDED
    )
    console.print(summary_panel)

def print_backtest_results(results):
    portfolio_values, trades, performance = results
    
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
    
    # Tabel untuk transaksi
    trade_table = Table(title="💹 Riwayat Transaksi", box=box.ROUNDED, border_style="bright_blue")
    trade_table.add_column("No", style="cyan", justify="center")
    trade_table.add_column("Hari", style="bright_blue")
    trade_table.add_column("Tipe", style="green")
    trade_table.add_column("Jumlah", justify="right", style="yellow")
    trade_table.add_column("Harga", justify="right", style="magenta")
    trade_table.add_column("Nilai", justify="right", style="bright_green")
    
    for i, trade in enumerate(trades[:10], 1):
        trade_table.add_row(
            str(i),
            str(trade['day']),
            trade['type'],
            f"{trade['shares']:.2f}",
            f"Rp {trade['price']:,.2f}",
            f"Rp {trade['value']:,.2f}"
        )
    
    if len(trades) > 10:
        trade_table.add_row("...", "...", "...", "...", "...", "...")
        trade_table.caption = f"Menampilkan 10 dari {len(trades)} transaksi"
    
    console.print(trade_table)

def save_results(predictor, y_true, y_pred, forecast, args):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        try:
            # Buat direktori results jika belum ada
            if not os.path.exists("results"):
                task = progress.add_task("Membuat direktori results...", total=1)
                os.makedirs("results")
                progress.update(task, advance=1)
            
            # Generate nama file berdasarkan ticker dan tanggal
            today = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/{args.ticker}_{today}"
            
            # Simpan plot
            task_plot = progress.add_task("Menyimpan plot...", total=1)
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
            progress.update(task_plot, advance=1)
            
            # Simpan hasil dalam CSV
            task_csv = progress.add_task("Menyimpan hasil ke CSV...", total=1)
            try:
                predictor.save_results_to_csv(f"{filename}_results.csv")
                progress.update(task_csv, advance=1)
                print_success(f"Hasil disimpan ke {filename}_results.csv")
            except Exception as e:
                print_warning(f"Gagal menyimpan hasil ke CSV: {str(e)}")
            
            print_success(f"Plot disimpan ke {filename}_plot.png")
        except Exception as e:
            print_warning(f"Gagal menyimpan hasil: {str(e)}")

def save_backtest_results(predictor, backtest_results, args):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        try:
            # Buat direktori results jika belum ada
            if not os.path.exists("results"):
                task = progress.add_task("Membuat direktori results...", total=1)
                os.makedirs("results")
                progress.update(task, advance=1)
            
            # Generate nama file
            today = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/{args.ticker}_{args.strategy}_{today}_backtest"
            
            portfolio_values, trades, performance = backtest_results
            
            # Simpan plot
            task_plot = progress.add_task("Menyimpan plot backtest...", total=1)
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_values, label='Nilai Portfolio', color='blue')
            plt.title(f'{args.ticker} - Backtest dengan {args.strategy}')
            plt.xlabel('Hari')
            plt.ylabel('Nilai Portfolio')
            plt.grid(True)
            plt.savefig(f"{filename}_plot.png", dpi=300, bbox_inches='tight')
            progress.update(task_plot, advance=1)
            
            # Simpan hasil backtest
            task_csv = progress.add_task("Menyimpan hasil backtest ke CSV...", total=1)
            predictor.save_backtest_results(f"{filename}_results.csv", 
                                         portfolio_values, trades, performance)
            progress.update(task_csv, advance=1)
            
            print_success(f"Hasil backtest disimpan ke {filename}_results.csv")
            print_success(f"Plot backtest disimpan ke {filename}_plot.png")
        except Exception as e:
            print_warning(f"Gagal menyimpan hasil backtest: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Aplikasi prediksi harga saham dengan PPO trading signals")
    
    # Parameter wajib
    parser.add_argument("--ticker", required=True, help="Kode saham (contoh: BMRI.JK)")
    parser.add_argument("--mode", required=True, choices=["predict", "backtest"], help="Mode operasi: predict atau backtest")
    
    # Parameter opsional
    parser.add_argument("--model", default="patchtst", choices=["patchtst"], help="Model prediksi yang digunakan")
    parser.add_argument("--start-date", help="Tanggal awal data (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Tanggal akhir data (YYYY-MM-DD)")
    parser.add_argument("--tune", action="store_true", help="Aktifkan hyperparameter tuning")
    parser.add_argument("--tuning-method", default="grid", choices=["grid", "hyperband"], help="Metode tuning hyperparameter")
    parser.add_argument("--log-dir", help="Direktori untuk menyimpan log")
    parser.add_argument("--ppo", action="store_true", help="Aktifkan PPO trading signals")
    parser.add_argument("--save-results", action="store_true", help="Simpan hasil prediksi")
    
    # Parameter lookback dan forecast
    parser.add_argument("--lookback", type=int, default=60, help="Jumlah hari historis yang digunakan untuk prediksi")
    parser.add_argument("--forecast-days", type=int, default=30, help="Jumlah hari yang akan diprediksi ke depan")
    
    # Parameter khusus backtest
    parser.add_argument("--strategy", default="PPO", choices=["Trend Following", "Mean Reversion", "Predictive", "PPO"], help="Strategi trading")
    parser.add_argument("--optimize", action="store_true", help="Aktifkan optimasi parameter strategi")
    parser.add_argument("--initial-balance", type=float, default=100000000, help="Modal awal untuk backtest")
    
    args = parser.parse_args()
    
    # Validasi parameter lookback dan forecast_days
    if args.lookback < 1:
        print_error("lookback harus lebih besar dari 0")
        sys.exit(1)
    
    if args.forecast_days < 1:
        print_error("forecast_days harus lebih besar dari 0")
        sys.exit(1)
    
    # Set tanggal default jika tidak diisi
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    return args

def main():
    print_header()
    
    # Buat layout untuk status
    layout = Layout()
    layout.split_column(
        Layout(name="header"),
        Layout(name="body"),
        Layout(name="footer")
    )
    
    args = parse_args()
    
    # Panel untuk informasi konfigurasi
    config_table = Table.grid(padding=1)
    config_table.add_column(style="bright_blue")
    config_table.add_column(style="bright_white")
    
    config_table.add_row("Ticker", args.ticker)
    config_table.add_row("Mode", args.mode.upper())
    config_table.add_row("Periode", f"{args.start_date} hingga {args.end_date}")
    config_table.add_row("Model", args.model.upper())
    if args.mode == 'backtest':
        config_table.add_row("Strategy", args.strategy)
        config_table.add_row("Initial Balance", f"Rp {args.initial_balance:,.2f}")
    config_table.add_row("PPO Trading Signals", "Aktif" if args.ppo else "Tidak Aktif")
    
    config_panel = Panel(
        config_table,
        title="⚙️ Konfigurasi",
        border_style="bright_blue",
        box=box.ROUNDED
    )
    console.print(config_panel)
    console.print()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            # Step 1: Initialize predictor
            predictor_task = progress.add_task("🚀 Memulai prediktor...", total=100)
            
            def update_progress(value, message):
                progress.update(predictor_task, completed=value, description=f"🚀 {message}")
            
            predictor = StockPredictor(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                model_type=args.model,
                lookback=args.lookback,
                forecast_days=args.forecast_days,
                tune_hyperparameters=args.tune,
                tuning_method=args.tuning_method,
                log_dir=args.log_dir,
                progress_callback=update_progress
            )
            
            # Step 2: Prepare data
            if not predictor.prepare_data():
                print_error("Gagal mempersiapkan data")
                return
            
            # Step 3: Train model
            predictor.train_model()
            
            # Step 4: Make predictions
            y_true, y_pred, forecast = predictor.predict()
            
            # Step 5: Evaluate model
            metrics = predictor.evaluate(y_true, y_pred)
            
            # Step 6: Generate trading signals if PPO is enabled
            trading_signals = []
            if args.ppo:
                ppo_task = progress.add_task("🤖 Menghasilkan sinyal trading dengan PPO...", total=100)
                
                # Get states for PPO
                states = predictor.get_states_for_ppo()
                progress.update(ppo_task, advance=30)
                
                # Initialize PPO agent with correct state dimension
                from src.trading.ppo_agent import PPOAgent
                ppo_agent = PPOAgent(
                    state_dim=5,  # Dimensi state dari get_states_for_ppo
                    action_dim=3,  # buy, sell, hold
                    hidden_dim=64
                )
                progress.update(ppo_task, advance=20)
                
                # Train PPO agent
                ppo_agent.train(states, n_epochs=50)
                progress.update(ppo_task, advance=30)
                
                # Generate trading signals
                for i in range(len(forecast)):
                    state = states[i] if i < len(states) else states[-1]
                    action, confidence = ppo_agent.get_action(state)
                    trading_signals.append({
                        'action': 'buy' if action == 0 else 'sell' if action == 1 else 'hold',
                        'confidence': confidence * 100
                    })
                progress.update(ppo_task, completed=100)
        
        # Print results
        print_model_metrics(metrics)
        
        if args.mode == 'predict':
            if args.ppo:
                print_forecast_with_signals(forecast, trading_signals)
                print_trading_summary(trading_signals)
            else:
                print_forecast(forecast)
                
            # Save prediction results if requested
            if args.save_results:
                save_results(predictor, y_true, y_pred, forecast, args)
                
        elif args.mode == 'backtest':
            # Run backtest
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                backtest_task = progress.add_task(f"📊 Menjalankan backtest dengan strategi {args.strategy}...", total=100)
                backtester = Backtester(
                    ticker=args.ticker,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    strategy=args.strategy,
                    initial_balance=args.initial_balance,
                    lookback=args.lookback,
                    forecast_days=args.forecast_days
                )
                
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
                    
                    # Optimasi
                    best_params, best_performance, best_portfolio, best_trades = optimizer.optimize(
                        args.strategy, param_ranges
                    )
                    
                    print_info(f"Parameter optimal: {best_params}")
                    backtest_results = (best_portfolio, best_trades, best_performance)
                else:
                    # Gunakan parameter default
                    backtest_results = backtester.run(args.strategy)
                
                progress.update(backtest_task, completed=100)
            
            # Print backtest results
            print_backtest_results(backtest_results)
            
            # Save backtest results if requested
            if args.save_results:
                save_backtest_results(predictor, backtest_results, args)
        
        console.print("\n✨ Selesai! ✨", style="bold green")
        
    except Exception as e:
        print_error(f"Terjadi kesalahan: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 