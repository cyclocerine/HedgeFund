#!/usr/bin/env python
"""
Pengujian Spesifik untuk AI Hedge Fund
=====================================
Skrip ini menjalankan pengujian spesifik untuk mengisolasi dan menemukan error.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Pastikan kita dapat mengimpor modul dari direktori utama
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading.risk_manager import RiskManager
from src.trading.portfolio import MultiAssetPortfolio

def test_risk_manager():
    """Uji RiskManager dengan berbagai skenario"""
    print("\n=== PENGUJIAN RISK MANAGER ===")
    
    # Inisialisasi risk manager
    risk_manager = RiskManager(
        max_drawdown=0.1,
        max_position_size=0.2,
        stop_loss=0.05,
        trailing_stop=0.03,
        max_capital_per_trade=0.2
    )
    
    # Uji position sizing
    position_size = risk_manager.size_position('BUY', 1000, 10000, 0.02)
    print(f"Position Size (volatilitas rendah): {position_size:.4f}")
    
    position_size = risk_manager.size_position('BUY', 1000, 10000, 0.05)
    print(f"Position Size (volatilitas tinggi): {position_size:.4f}")
    
    # Uji stop loss detection
    positions = {
        'AAPL': {
            'type': 'LONG',
            'entry_price': 150,
            'shares': 10,
            'high_price': 160
        },
        'MSFT': {
            'type': 'SHORT',
            'entry_price': 300,
            'shares': 5,
            'low_price': 290
        }
    }
    
    # Uji harga di bawah stop loss
    current_prices = {'AAPL': 142, 'MSFT': 315}
    signals = risk_manager.check_stop_loss(positions, current_prices)
    print(f"Stop Loss Signals: {signals}")
    
    # Uji trailing stop
    current_prices = {'AAPL': 155, 'MSFT': 300}
    positions['AAPL']['high_price'] = 165  # Update high price
    signals = risk_manager.check_stop_loss(positions, current_prices)
    print(f"Trailing Stop Signals: {signals}")
    
    # Uji portfolio risk check
    portfolio_value = 9000  # 10% drawdown dari high
    risk_action = risk_manager.check_portfolio_risk(portfolio_value, positions)
    print(f"Portfolio Risk Action: {risk_action}")
    
    return True

def test_portfolio():
    """Uji portfolio multi-aset"""
    print("\n=== PENGUJIAN PORTFOLIO MULTI-ASSET SEDERHANA ===")
    
    # Inisialisasi portfolio
    portfolio = MultiAssetPortfolio(
        assets=['AAPL', 'MSFT'],
        initial_capital=10000,
        transaction_fee=0.001
    )
    
    # Set risk manager
    risk_manager = RiskManager(
        max_drawdown=0.1,
        max_position_size=0.2,
        stop_loss=0.05,
        trailing_stop=0.03,
        max_capital_per_trade=0.2
    )
    portfolio.risk_manager = risk_manager
    
    # Simulasi 5 hari trading
    start_date = datetime.now()
    
    # Harga awal
    prices = {
        'AAPL': 150,
        'MSFT': 300
    }
    
    # Update harga awal
    portfolio.update_prices(prices, timestamp=start_date)
    
    # Buat signal untuk membeli AAPL
    signals = [{
        'symbol': 'AAPL',
        'action': 'BUY',
        'size': 0.2,
        'volatility': 0.02
    }]
    
    # Alokasi modal hari 1
    result = portfolio.allocate_capital(signals, prices, timestamp=start_date + timedelta(days=1))
    print(f"Hari 1 - Beli AAPL: {result['executed_orders']}")
    
    # Update harga hari 2 (AAPL naik)
    prices['AAPL'] = 155
    portfolio.update_prices(prices, timestamp=start_date + timedelta(days=2))
    
    # Buat signal untuk membeli MSFT
    signals = [{
        'symbol': 'MSFT',
        'action': 'BUY',
        'size': 0.2,
        'volatility': 0.015
    }]
    
    # Alokasi modal hari 3
    result = portfolio.allocate_capital(signals, prices, timestamp=start_date + timedelta(days=3))
    print(f"Hari 3 - Beli MSFT: {result['executed_orders']}")
    
    # Update harga hari 4 (MSFT turun)
    prices['MSFT'] = 290
    portfolio.update_prices(prices, timestamp=start_date + timedelta(days=4))
    
    # Update harga hari 5 (AAPL naik lagi)
    prices['AAPL'] = 160
    portfolio.update_prices(prices, timestamp=start_date + timedelta(days=5))
    
    # Buat signal untuk menjual AAPL
    signals = [{
        'symbol': 'AAPL',
        'action': 'SELL',
        'size': 1.0,
        'volatility': 0.02
    }]
    
    # Alokasi modal hari 6
    result = portfolio.allocate_capital(signals, prices, timestamp=start_date + timedelta(days=6))
    print(f"Hari 6 - Jual AAPL: {result['executed_orders']}")
    
    # Dapatkan ringkasan portfolio
    summary = portfolio.get_portfolio_summary()
    print(f"\nRingkasan Portfolio:")
    print(f"Nilai Awal: Rp {portfolio.initial_capital:,.2f}")
    print(f"Nilai Akhir: Rp {summary['current_value']:,.2f}")
    print(f"Return: {summary['return_pct']:.2f}%")
    print(f"Cash: Rp {summary['cash']:,.2f}")
    
    # Dapatkan daftar transaksi
    transactions_df = portfolio.get_transactions_df()
    print(f"\nJumlah Transaksi: {len(transactions_df)}")
    
    # Dapatkan posisi aktif
    positions_df = portfolio.get_positions_df()
    print(f"\nPosisi Aktif:")
    if len(positions_df) == 0:
        print("Tidak ada posisi aktif")
    else:
        print(positions_df)
        
    return True

if __name__ == "__main__":
    try:
        test_risk_manager()
        print("\nPengujian RiskManager berhasil!")
    except Exception as e:
        print(f"\nError pada pengujian RiskManager: {str(e)}")
    
    try:
        test_portfolio()
        print("\nPengujian Portfolio berhasil!")
    except Exception as e:
        print(f"\nError pada pengujian Portfolio: {str(e)}") 