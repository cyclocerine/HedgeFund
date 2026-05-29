#!/usr/bin/env python
"""
Walk-Forward Optimization Backtest
===================================
Script untuk menjalankan Walk-Forward Optimization (Rolling Window).
Mencegah Concept Drift dengan melatih ulang model secara periodik.
"""

import argparse
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predictor import StockPredictor
from src.trading.backtest import Backtester

def parse_args():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization Backtest")
    parser.add_argument('--ticker', type=str, default='NVDA', help='Stock Ticker')
    parser.add_argument('--start-date', type=str, default='2021-01-01', help='Global start date')
    parser.add_argument('--end-date', type=str, default='2024-01-01', help='Global end date')
    parser.add_argument('--train-window', type=int, default=252, help='Train window in days')
    parser.add_argument('--test-window', type=int, default=63, help='Test window in days')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per window (keep low for speed)')
    parser.add_argument('--model', type=str, default='plstm', choices=['patchtst', 'improved_patchtst', 'plstm', 'bilstm'])
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print(f" WALK-FORWARD OPTIMIZATION: {args.ticker}")
    print(f" Model: {args.model} | Train: {args.train_window} days | Test: {args.test_window} days")
    print("=" * 60)
    
    # 1. Unduh dataset master untuk referensi tanggal
    print("Mengunduh dataset master...")
    master_predictor = StockPredictor(args.ticker, args.start_date, args.end_date, model_type=args.model)
    if not master_predictor.preprocessor.download_data():
        print("Gagal mengunduh data.")
        return
        
    master_dates = master_predictor.preprocessor.data.index
    total_days = len(master_dates)
    
    if total_days <= args.train_window:
        print(f"Error: Total data ({total_days} hari) lebih pendek dari train window ({args.train_window} hari)")
        return
        
    print(f"Total Trading Days: {total_days}")
    
    current_idx = args.train_window
    window_count = 1
    
    all_actual = []
    all_predicted = []
    all_test_dates = []
    
    while current_idx < total_days:
        test_end_idx = min(current_idx + args.test_window, total_days)
        test_len = test_end_idx - current_idx
        
        train_start_date = master_dates[current_idx - args.train_window].strftime('%Y-%m-%d')
        train_end_date = master_dates[current_idx - 1].strftime('%Y-%m-%d')
        
        test_start_date = master_dates[current_idx].strftime('%Y-%m-%d')
        test_end_date = master_dates[test_end_idx - 1].strftime('%Y-%m-%d')
        
        print(f"\n--- WFO WINDOW {window_count} ---")
        print(f"Train : {train_start_date} s/d {train_end_date} ({args.train_window} hari)")
        print(f"Test  : {test_start_date} s/d {test_end_date} ({test_len} hari)")
        
        # Inisialisasi predictor untuk window ini (mencakup train + test)
        window_predictor = StockPredictor(
            args.ticker, 
            train_start_date, 
            test_end_date, 
            model_type=args.model,
            lookback=60,
            forecast_days=1 # Iterasi
        )
        
        print("Menyiapkan data window...")
        # Proporsikan train_ratio secara eksak agar model stop belajar sebelum test window
        total_window_len = args.train_window + test_len
        train_ratio = args.train_window / total_window_len
        
        success = window_predictor.prepare_data(train_ratio=train_ratio)
        if not success:
            print("Gagal menyiapkan data untuk window ini. Skip.")
            break
            
        print("Melatih model (Deep Learning)...")
        window_predictor.train_model(epochs=args.epochs, batch_size=32)
        
        print("Memprediksi window test...")
        y_true, y_pred, _ = window_predictor.predict()
        
        # Ambil hanya bagian test
        test_actual = y_true[-test_len:]
        test_predicted = y_pred[-test_len:]
        
        all_actual.extend(test_actual)
        all_predicted.extend(test_predicted)
        all_test_dates.extend(master_dates[current_idx:test_end_idx])
        
        current_idx += args.test_window
        window_count += 1
        
    print("\n" + "=" * 60)
    print("WFO SELESAI. Mengeksekusi Backtest Global...")
    print("=" * 60)
    
    # Jalankan simulasi trading PADA KESELURUHAN HASIL WFO
    # Di sini Slippage Dinamis yang diimplementasi akan teruji secara otomatis!
    backtester = Backtester(
        actual_prices=all_actual,
        predicted_prices=all_predicted,
        initial_investment=100000.0,
        transaction_fee=0.001,
        dates=all_test_dates
    )
    
    portfolio_values, trades, performance = backtester.run(strategy='Predictive', allow_short=True)
    
    print("\nPerforma Keseluruhan WFO (termasuk Slippage Dinamis):")
    print(f"Initial Value: ${performance['initial_investment']:,.2f}")
    print(f"Final Value:   ${performance['final_value']:,.2f}")
    print(f"Total Return:  {performance['total_return']:.2f}%")
    print(f"Max Drawdown:  {performance['max_drawdown']:.2f}%")
    print(f"Win Rate:      {performance['win_rate']:.2f}%")
    
if __name__ == "__main__":
    main()
