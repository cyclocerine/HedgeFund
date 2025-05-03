#!/usr/bin/env python
"""
CLI Application
=============

Script untuk menjalankan aplikasi prediksi saham dengan antarmuka command line.
"""

import argparse
from datetime import datetime
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predictor import StockPredictor
from src.trading.backtest import Backtester
from src.trading.optimizer import StrategyOptimizer

def print_header():
    print("=" * 60)
    print("  🚀 APLIKASI PREDIKSI HARGA SAHAM - MODE CLI")
    print("  Powered by Deep Learning Models")
    print("=" * 60)
    print()

def print_step(step_number, total_steps, step_name):
    print(f"[{step_number}/{total_steps}] 🔄 {step_name}...")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ ERROR: {message}")

def print_warning(message):
    print(f"⚠️ PERINGATAN: {message}")

def print_info(message):
    print(f"ℹ️ {message}")

def print_model_metrics(metrics):
    print("\n📊 Metrik Evaluasi Model:")
    print("-" * 40)
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")
    print("-" * 40)

def print_forecast(forecast):
    print("\n🔮 Prediksi untuk hari-hari berikutnya:")
    print("-" * 40)
    for i, price in enumerate(forecast, 1):
        # Tambahkan indikator tren jika ada nilai sebelumnya
        trend = ""
        if i > 1:
            prev_price = forecast[i-2]
            pct_change = (price - prev_price) / prev_price * 100
            if pct_change > 0:
                trend = f"(↗️ +{pct_change:.2f}%)"
            else:
                trend = f"(↘️ {pct_change:.2f}%)"
        
        print(f"Hari {i}: Rp {price:.2f} {trend}")
    print("-" * 40)

def print_backtest_results(results):
    """
    Mencetak hasil backtest ke console
    
    Parameters:
    -----------
    results : tuple
        (portfolio_values, trades, performance)
    """
    portfolio_values, trades, performance = results
    
    print("\n📈 Hasil Backtesting:")
    print("-" * 40)
    print(f"Investasi Awal: Rp {performance['initial_investment']:.2f}")
    print(f"Nilai Akhir: Rp {performance['final_value']:.2f}")
    print(f"Return Total: {performance['total_return']:.2f}%")
    print(f"Maximum Drawdown: {performance['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
    print(f"Win Rate: {performance['win_rate']:.2f}%")
    print(f"Jumlah Transaksi: {performance['num_trades']}")
    print("-" * 40)
    
    print("\n💹 Transaksi:")
    print("-" * 60)
    for i, trade in enumerate(trades[:10]):  # Hanya tampilkan 10 transaksi pertama
        print(f"{i+1}. Hari {trade['day']}: {trade['type']} - {trade['shares']:.2f} saham " + 
              f"@ Rp {trade['price']:.2f} = Rp {trade['value']:.2f}")
    
    if len(trades) > 10:
        print(f"... dan {len(trades) - 10} transaksi lainnya")
    print("-" * 60)

def save_results(predictor, y_true, y_pred, forecast, args):
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
        
        # Simpan hasil dalam CSV
        try:
            predictor.save_results_to_csv(f"{filename}_results.csv")
            print_success(f"Hasil disimpan ke {filename}_results.csv")
        except Exception as e:
            print_warning(f"Gagal menyimpan hasil ke CSV: {str(e)}")
        
        print_success(f"Plot disimpan ke {filename}_plot.png")
    except Exception as e:
        print_warning(f"Gagal menyimpan hasil: {str(e)}")

def save_backtest_results(predictor, backtest_results, args):
    """
    Simpan hasil backtest ke file
    
    Parameters:
    -----------
    predictor : StockPredictor
        Objek prediktor yang digunakan
    backtest_results : tuple
        (portfolio_values, trades, performance)
    args : argparse.Namespace
        Argumen command line
    """
    try:
        # Buat direktori results jika belum ada
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # Generate nama file berdasarkan ticker dan tanggal
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
        
        # Simpan hasil backtest
        predictor.save_backtest_results(f"{filename}_results.csv", 
                                        portfolio_values, trades, performance)
        
        print_success(f"Hasil backtest disimpan ke {filename}_results.csv")
        print_success(f"Plot backtest disimpan ke {filename}_plot.png")
    except Exception as e:
        print_warning(f"Gagal menyimpan hasil backtest: {str(e)}")

def main():
    print_header()
    
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--ticker', type=str, default='ADRO.JK', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--lookback', type=int, default=60, help='Number of days to look back')
    parser.add_argument('--forecast_days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--model', type=str, default='ensemble', 
                        choices=['cnn_lstm', 'bilstm', 'transformer', 'ensemble'], 
                        help='Model type')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--save', action='store_true', help='Save results to files')
    
    # Tambahkan argumen untuk backtest dan strategi
    parser.add_argument('--backtest', action='store_true', help='Run backtest on historical data')
    parser.add_argument('--strategy', type=str, default='Predictive', 
                       choices=['Trend Following', 'Mean Reversion', 'Predictive', 'PPO'],
                       help='Trading strategy to use')
    parser.add_argument('--optimize', action='store_true', help='Optimize strategy parameters')
    
    args = parser.parse_args()
    
    print_info(f"Ticker: {args.ticker}")
    print_info(f"Periode: {args.start_date} hingga {args.end_date}")
    print_info(f"Model: {args.model.upper()}")
    print_info(f"Lookback: {args.lookback} hari, Forecast: {args.forecast_days} hari")
    print_info(f"Hyperparameter Tuning: {'Aktif' if args.tune else 'Tidak Aktif'}")
    if args.backtest:
        print_info(f"Backtest: Aktif, Strategi: {args.strategy}")
        print_info(f"Optimisasi Parameter: {'Aktif' if args.optimize else 'Tidak Aktif'}")
    print()
    
    try:
        # Step 1: Initialize predictor
        print_step(1, 5 if not args.backtest else 7, "Memulai prediktor dan mengunduh data")
        predictor = StockPredictor(
            args.ticker,
            args.start_date,
            args.end_date,
            args.lookback,
            args.forecast_days,
            args.model,
            args.tune
        )
        
        # Step 2: Prepare data
        print_step(2, 5 if not args.backtest else 7, "Mempersiapkan data")
        if not predictor.prepare_data():
            print_error("Gagal mempersiapkan data")
            return
        print_success("Data berhasil dipersiapkan")
        
        # Step 3: Train model
        print_step(3, 5 if not args.backtest else 7, f"Melatih model {args.model.upper()}")
        start_time = time.time()
        history = predictor.train_model()
        training_time = time.time() - start_time
        print_success(f"Model berhasil dilatih dalam {training_time:.2f} detik")
        
        # Step 4: Make predictions
        print_step(4, 5 if not args.backtest else 7, "Membuat prediksi")
        y_true, y_pred, forecast = predictor.predict()
        print_success("Prediksi berhasil dibuat")
        
        # Step 5: Evaluate model
        print_step(5, 5 if not args.backtest else 7, "Mengevaluasi model")
        metrics = predictor.evaluate(y_true, y_pred)
        print_success("Evaluasi selesai")
        
        # Print results
        print_model_metrics(metrics)
        print_forecast(forecast)
        
        # Jalankan backtest jika diminta
        if args.backtest:
            # Step 6: Run backtest
            print_step(6, 7, f"Menjalankan backtest dengan strategi {args.strategy}")
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
                        'episodes': [5, 10]  # Untuk CLI gunakan episodes yang lebih kecil
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
            
            print_success("Backtest selesai")
            
            # Print backtest results
            print_backtest_results(backtest_results)
            
            # Step 7: Save backtest results if requested
            if args.save:
                print_step(7, 7, "Menyimpan hasil backtest")
                save_backtest_results(predictor, backtest_results, args)
        else:
            # Plot and save results if requested
            if args.save:
                print_info("Menyimpan hasil...")
                save_results(predictor, y_true, y_pred, forecast, args)
        
        print("\n✨ Prediksi selesai! ✨")
        
    except Exception as e:
        print_error(f"Terjadi kesalahan: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 