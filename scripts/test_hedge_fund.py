#!/usr/bin/env python
"""
Test Hedge Fund Components
=========================

Script untuk menguji komponen-komponen AI Hedge Fund.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading.risk_manager import RiskManager
from src.trading.portfolio import MultiAssetPortfolio
from src.trading.backtest import Backtester

def test_risk_manager():
    """Menguji fungsi RiskManager"""
    print("\n=== Menguji Risk Manager ===")
    
    # Inisialisasi risk manager
    risk_manager = RiskManager(
        max_drawdown=0.1,
        max_position_size=0.2,
        stop_loss=0.05,
        trailing_stop=0.03
    )
    
    # Test portfolio risk check
    portfolio_value = 100000
    positions = {
        'AAPL': {
            'type': 'LONG',
            'entry_price': 150,
            'shares': 10,
            'high_price': 160,
            'current_value': 1550
        },
        'MSFT': {
            'type': 'SHORT',
            'entry_price': 300,
            'shares': 5,
            'low_price': 290,
            'current_value': 1450
        }
    }
    
    # Check portfolio risk
    risk_action = risk_manager.check_portfolio_risk(portfolio_value, positions)
    print(f"Portfolio Risk Check: {risk_action}")
    
    # Test position sizing
    position_size = risk_manager.size_position('BUY', 150, 100000, 0.02)
    print(f"Position Sizing: {position_size}")
    
    # Test stop loss check
    current_prices = {'AAPL': 142, 'MSFT': 310}
    stop_signals = risk_manager.check_stop_loss(positions, current_prices)
    print(f"Stop Loss Signals: {stop_signals}")
    
    # Test portfolio metrics
    portfolio_history = [100000, 101000, 99000, 102000, 103000, 101500]
    metrics = risk_manager.calculate_portfolio_metrics(portfolio_history)
    print(f"Portfolio Metrics: {metrics}")

def test_portfolio():
    """Menguji fungsi MultiAssetPortfolio"""
    print("\n=== Menguji Multi-Asset Portfolio ===")
    
    # Inisialisasi portfolio
    portfolio = MultiAssetPortfolio(
        assets=['AAPL', 'MSFT', 'BTC'],
        initial_capital=100000,
        transaction_fee=0.001
    )
    
    # Buat data harga dan signals
    current_prices = {
        'AAPL': 150.25,
        'MSFT': 270.50,
        'BTC': 28950.75
    }
    
    signals = [
        {
            'symbol': 'AAPL',
            'action': 'BUY',
            'size': 0.2,
            'volatility': 0.015
        },
        {
            'symbol': 'BTC',
            'action': 'SHORT',
            'size': 0.1,
            'volatility': 0.045
        }
    ]
    
    # Alokasikan kapital berdasarkan sinyal
    timestamp = datetime.now()
    result = portfolio.allocate_capital(signals, current_prices, timestamp)
    print(f"Alokasi Kapital: {result}")
    
    # Update harga dan lihat nilai portofolio
    new_prices = {
        'AAPL': 152.30,
        'MSFT': 275.20,
        'BTC': 28500.50
    }
    
    new_timestamp = timestamp + timedelta(days=1)
    portfolio_value = portfolio.update_prices(new_prices, new_timestamp)
    print(f"Nilai Portofolio: {portfolio_value}")
    
    # Generate sinyal baru untuk menutup posisi
    close_signals = [
        {
            'symbol': 'AAPL',
            'action': 'SELL',
            'size': 0.5,  # Jual setengah posisi
            'volatility': 0.015
        },
        {
            'symbol': 'BTC',
            'action': 'COVER',
            'size': 1.0,  # Tutup seluruh posisi short
            'volatility': 0.045
        }
    ]
    
    # Alokasikan kapital untuk menutup posisi
    result = portfolio.allocate_capital(close_signals, new_prices, new_timestamp + timedelta(days=1))
    print(f"Hasil Menutup Posisi: {result}")
    
    # Dapatkan ringkasan portofolio
    summary = portfolio.get_portfolio_summary()
    print(f"Ringkasan Portofolio: {summary}")
    
    # Dapatkan posisi aktif
    positions = portfolio.get_positions_df()
    print(f"Posisi Aktif:\n{positions}")
    
    # Dapatkan history transaksi
    transactions = portfolio.get_transactions_df()
    print(f"Transaksi:\n{transactions}")

def test_backtester():
    """Menguji fungsi Backtester dengan fitur baru"""
    print("\n=== Menguji Enhanced Backtester ===")
    
    # Generate sample data
    np.random.seed(42)
    days = 100
    
    # Generate random price series with trend
    actual_prices = np.cumsum(np.random.normal(0.001, 0.02, days)) + 10
    
    # Generate predicted prices with some noise
    predicted_prices = actual_prices + np.random.normal(0, 0.5, days)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Initialize backtester
    backtester = Backtester(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        initial_investment=10000,
        transaction_fee=0.001,
        dates=dates
    )
    
    # Run backtest with short selling enabled
    print("Running backtest with short selling...")
    portfolio_values, trades, performance = backtester.run(
        strategy="Predictive",
        allow_short=True,
        max_position_size=0.5
    )
    
    print(f"Backtest Performance: {performance}")
    print(f"Number of Trades: {len(trades)}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Perbaiki masalah plot dengan menyesuaikan ukuran dates dengan portfolio_values
    plt_dates = dates[:len(portfolio_values)]
    plt.plot(plt_dates, portfolio_values, label='Portfolio Value')
    
    plt.title('Backtest Results with Enhanced Backtester')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Simpan plot ke file
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/backtest_test_result.png", dpi=300)
    print("Plot saved to results/backtest_test_result.png")

if __name__ == "__main__":
    print("=== AI Hedge Fund Component Test ===")
    
    try:
        test_risk_manager()
        test_portfolio()
        test_backtester()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during tests: {str(e)}")
        import traceback
        traceback.print_exc() 