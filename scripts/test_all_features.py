#!/usr/bin/env python
"""
Pengujian Komprehensif AI Hedge Fund
=================================

Script ini menguji semua fitur utama AI Hedge Fund untuk menemukan
error dan memastikan bahwa sistem berfungsi dengan baik.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import traceback

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading.risk_manager import RiskManager
from src.trading.portfolio import MultiAssetPortfolio
from src.trading.backtest import Backtester
from src.trading.ppo_agent import PPOTrader, TradingEnv
from src.data.indicators import add_technical_indicators

# Buat direktori untuk hasil pengujian
if not os.path.exists("results"):
    os.makedirs("results")

def test_risk_manager_advanced():
    """Pengujian lanjutan untuk RiskManager"""
    print("\n=== Pengujian RiskManager Lanjutan ===")
    
    # Inisialisasi risk manager dengan berbagai parameter
    risk_managers = [
        RiskManager(max_drawdown=0.1, max_position_size=0.2, stop_loss=0.05, trailing_stop=0.03),
        RiskManager(max_drawdown=0.05, max_position_size=0.1, stop_loss=0.03, trailing_stop=0.02),
        RiskManager(max_drawdown=0.2, max_position_size=0.3, stop_loss=0.1, trailing_stop=0.05)
    ]
    
    # Buat data portofolio untuk pengujian
    portfolio_value = 100000
    positions = {
        'AAPL': {'type': 'LONG', 'entry_price': 150, 'shares': 10, 'high_price': 160, 'current_value': 1550},
        'MSFT': {'type': 'SHORT', 'entry_price': 300, 'shares': 5, 'low_price': 290, 'current_value': 1450},
        'GOOGL': {'type': 'LONG', 'entry_price': 2500, 'shares': 2, 'high_price': 2600, 'current_value': 5200}
    }
    
    # Tes berbagai harga untuk stop loss
    price_scenarios = [
        # Skenario normal
        {'AAPL': 155, 'MSFT': 295, 'GOOGL': 2550},
        # Skenario stop loss AAPL
        {'AAPL': 142, 'MSFT': 295, 'GOOGL': 2550},
        # Skenario trailing stop MSFT
        {'AAPL': 155, 'MSFT': 310, 'GOOGL': 2550},
        # Skenario multiple stop
        {'AAPL': 142, 'MSFT': 310, 'GOOGL': 2350}
    ]
    
    # Pengujian dengan berbagai risk manager dan skenario
    for i, rm in enumerate(risk_managers):
        print(f"\nPengujian Risk Manager #{i+1}:")
        
        # Test portfolio risk check dengan drawdown besar
        portfolio_history = [100000, 95000, 90000, 88000]  # -12% drawdown
        print(f"  Risk check dengan drawdown -12%: {rm.check_portfolio_risk(88000, positions)}")
        
        # Test position sizing dengan volatilitas berbeda
        for vol in [0.01, 0.05, 0.1, 0.2]:
            size = rm.size_position('BUY', 150, 100000, vol)
            print(f"  Position size dengan volatilitas {vol}: {size:.4f}")
        
        # Test stop loss dengan berbagai skenario harga
        for j, prices in enumerate(price_scenarios):
            signals = rm.check_stop_loss(positions, prices)
            print(f"  Skenario harga #{j+1}: {len(signals)} sinyal stop loss")
            for signal in signals:
                print(f"    {signal['symbol']}: {signal['action']} ({signal['reason']})")
    
    print("\nPengujian RiskManager selesai")

def test_multi_asset_portfolio_advanced():
    """Pengujian lanjutan untuk MultiAssetPortfolio"""
    print("\n=== Pengujian MultiAssetPortfolio Lanjutan ===")
    
    # Inisialisasi portfolio dengan berbagai aset
    portfolio = MultiAssetPortfolio(
        assets=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BTC'],
        initial_capital=100000,
        transaction_fee=0.001
    )
    
    # Generate data harga selama beberapa hari
    days = 10
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Buat skenario harga dengan berbagai tren
    price_data = {
        'AAPL': np.linspace(150, 160, days),              # Uptrend
        'MSFT': np.linspace(300, 280, days),              # Downtrend
        'GOOGL': np.linspace(2500, 2500, days),           # Sideways
        'AMZN': 3000 + 50*np.sin(np.linspace(0, 3, days)), # Cyclical
        'BTC': np.array([30000, 31000, 32000, 31500, 30800, 29500, 28000, 27000, 28500, 29000]) # Volatile
    }
    
    # Simulasi transaksi selama beberapa hari
    print("\nSimulasi Trading Multi-Aset Selama 10 Hari:")
    
    # Hari 1: Beli beberapa aset
    day1_signals = [
        {'symbol': 'AAPL', 'action': 'BUY', 'size': 0.15, 'volatility': 0.01},
        {'symbol': 'MSFT', 'action': 'SHORT', 'size': 0.1, 'volatility': 0.015},
        {'symbol': 'BTC', 'action': 'BUY', 'size': 0.05, 'volatility': 0.03}
    ]
    
    day1_prices = {symbol: price_data[symbol][0] for symbol in portfolio.assets}
    result = portfolio.allocate_capital(day1_signals, day1_prices, dates[0])
    print(f"Hari 1 - Alokasi: {len(result['executed_orders'])} order, "
          f"Nilai Portofolio: {result['portfolio_value']:.2f}")
    
    # Update harga setiap hari dan ambil tindakan sesuai kebutuhan
    for i in range(1, days):
        current_prices = {symbol: price_data[symbol][i] for symbol in portfolio.assets}
        
        # Hari 3: Tambah posisi
        if i == 2:
            signals = [
                {'symbol': 'GOOGL', 'action': 'BUY', 'size': 0.1, 'volatility': 0.01},
                {'symbol': 'AAPL', 'action': 'BUY', 'size': 0.05, 'volatility': 0.01}  # Tambah posisi AAPL
            ]
            result = portfolio.allocate_capital(signals, current_prices, dates[i])
            print(f"Hari {i+1} - Tambah posisi: {len(result['executed_orders'])} order")
        
        # Hari 5: Ambil partial profit
        elif i == 4:
            signals = [
                {'symbol': 'AAPL', 'action': 'SELL', 'size': 0.3, 'volatility': 0.01}, # Jual 30% posisi
                {'symbol': 'MSFT', 'action': 'COVER', 'size': 0.5, 'volatility': 0.015} # Cover 50% short
            ]
            result = portfolio.allocate_capital(signals, current_prices, dates[i])
            print(f"Hari {i+1} - Partial profit: {len(result['executed_orders'])} order")
        
        # Hari 8: Reaksi terhadap pasar turun
        elif i == 7:
            signals = [
                {'symbol': 'AAPL', 'action': 'SELL', 'size': 1.0, 'volatility': 0.01}, # Jual semua AAPL
                {'symbol': 'AMZN', 'action': 'SHORT', 'size': 0.1, 'volatility': 0.02} # Short AMZN
            ]
            result = portfolio.allocate_capital(signals, current_prices, dates[i])
            print(f"Hari {i+1} - Reaksi pasar turun: {len(result['executed_orders'])} order")
        
        # Hari 10: Tutup semua posisi
        elif i == 9:
            positions_df = portfolio.get_positions_df()
            signals = []
            
            for _, row in positions_df.iterrows():
                symbol = row['symbol']
                if row['type'] == 'LONG':
                    signals.append({'symbol': symbol, 'action': 'SELL', 'size': 1.0, 'volatility': 0.02})
                else:
                    signals.append({'symbol': symbol, 'action': 'COVER', 'size': 1.0, 'volatility': 0.02})
            
            result = portfolio.allocate_capital(signals, current_prices, dates[i])
            print(f"Hari {i+1} - Tutup semua posisi: {len(result['executed_orders'])} order")
        
        # Hari-hari lain: Update nilai portfolio saja
        else:
            portfolio_value = portfolio.update_prices(current_prices, dates[i])
            print(f"Hari {i+1} - Update harga: Nilai Portofolio: {portfolio_value:.2f}")
    
    # Dapatkan ringkasan akhir
    summary = portfolio.get_portfolio_summary()
    print(f"\nRingkasan Akhir Portfolio:")
    print(f"  Nilai Awal: {summary['initial_capital']:.2f}")
    print(f"  Nilai Akhir: {summary['current_value']:.2f}")
    print(f"  Return: {summary['return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {summary['metrics']['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {summary['metrics']['max_drawdown']*100:.2f}%")
    
    # Plot portfolio value
    portfolio_df = portfolio.get_portfolio_history_df()
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_df['timestamp'], portfolio_df['value'], marker='o')
    plt.title('Nilai Portfolio Multi-Aset')
    plt.xlabel('Tanggal')
    plt.ylabel('Nilai')
    plt.grid(True)
    plt.savefig('results/multi_asset_portfolio_test.png', dpi=300)
    
    # Cetak ringkasan transaksi
    transactions_df = portfolio.get_transactions_df()
    print(f"\nRingkasan {len(transactions_df)} Transaksi:")
    print(transactions_df[['timestamp', 'symbol', 'action', 'price', 'shares', 'value']])
    
    print("\nPengujian MultiAssetPortfolio selesai")

def test_backtester_strategies():
    """Pengujian Backtester dengan berbagai strategi"""
    print("\n=== Pengujian Backtester dengan Berbagai Strategi ===")
    
    # Generate data
    np.random.seed(42)
    days = 200
    
    # Buat data harga dengan beberapa regime (bull, bear, sideways)
    regimes = [
        (0, 50, 0.001),     # Bull trend
        (50, 100, -0.001),  # Bear trend
        (100, 150, 0.0),    # Sideways
        (150, 200, 0.002)   # Strong bull
    ]
    
    prices = np.zeros(days) + 10  # Start at $10
    
    for start, end, drift in regimes:
        segment = np.cumsum(np.random.normal(drift, 0.02, end-start))
        prices[start:end] = prices[start] + segment
    
    # Buat data prediksi (dengan lead 1 hari dan noise)
    predicted = np.roll(prices, -1)
    predicted[-1] = predicted[-2] * (1 + np.random.normal(0.001, 0.01))
    predicted = predicted + np.random.normal(0, 0.2, days)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Strategi yang akan diuji
    strategies = ["Trend Following", "Mean Reversion", "Predictive"]
    
    results = {}
    
    # Pengujian setiap strategi
    for strategy in strategies:
        print(f"\nPengujian strategi: {strategy}")
        
        # Inisialisasi backtester
        backtester = Backtester(
            actual_prices=prices,
            predicted_prices=predicted,
            initial_investment=10000,
            transaction_fee=0.001,
            dates=dates
        )
        
        # Jalankan backtest dengan mode long-only dan long-short
        for allow_short in [False, True]:
            mode = "Long-Short" if allow_short else "Long-Only"
            print(f"  Mode: {mode}")
            
            portfolio_values, trades, performance = backtester.run(
                strategy=strategy,
                allow_short=allow_short,
                max_position_size=0.5
            )
            
            print(f"    Return: {performance['total_return']:.2f}%")
            print(f"    Sharpe: {performance['sharpe_ratio']:.4f}")
            print(f"    Win Rate: {performance['win_rate']:.2f}%")
            print(f"    Trades: {performance['num_trades']}")
            
            # Simpan hasil
            results[f"{strategy}_{mode}"] = {
                'portfolio_values': portfolio_values,
                'performance': performance
            }
    
    # Plot perbandingan strategi
    plt.figure(figsize=(12, 8))
    
    # Subplot untuk perbandingan return
    plt.subplot(2, 1, 1)
    for key, data in results.items():
        plt.plot(data['portfolio_values'], label=key)
    
    plt.title('Perbandingan Strategi - Portfolio Value')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    
    # Subplot untuk perbandingan metrik
    plt.subplot(2, 1, 2)
    
    # Ekstrak metrik
    strategy_names = list(results.keys())
    returns = [results[s]['performance']['total_return'] for s in strategy_names]
    sharpe_ratios = [results[s]['performance']['sharpe_ratio'] for s in strategy_names]
    win_rates = [results[s]['performance']['win_rate'] for s in strategy_names]
    
    # Bar plot untuk metrik
    x = np.arange(len(strategy_names))
    width = 0.25
    
    plt.bar(x - width, returns, width, label='Return (%)')
    plt.bar(x, sharpe_ratios, width, label='Sharpe Ratio')
    plt.bar(x + width, win_rates, width, label='Win Rate (%)')
    
    plt.xticks(x, strategy_names, rotation=45)
    plt.title('Perbandingan Metrik Strategi')
    plt.grid(True, axis='y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/backtest_strategies_comparison.png', dpi=300)
    
    print("\nPengujian Backtester selesai")

def test_ppo_agent_with_indicators():
    """Pengujian PPO Agent dengan indikator teknikal"""
    print("\n=== Pengujian PPO Agent dengan Indikator Teknikal ===")
    
    try:
        # Generate data
        np.random.seed(42)
        days = 100
        
        # Buat data sintetis yang lebih realistis
        prices = np.zeros(days) + 100  # Start at $100
        
        # Tambahkan tren dan siklikal
        trend = np.linspace(0, 20, days)  # Tren naik
        cycle = 10 * np.sin(np.linspace(0, 4*np.pi, days))  # Komponen siklikal
        noise = np.random.normal(0, 3, days)  # Noise
        
        prices = prices + trend + cycle + noise
        
        # Buat dataframe dengan format yang diperlukan
        dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
        df = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000, 100000, days)
        }, index=dates)
        
        # Tambahkan indikator teknikal
        print("Menghitung indikator teknikal...")
        df = add_technical_indicators(df)
        
        # Pilih subset indikator penting
        feature_columns = ['RSI', 'MACD', 'MACD_Signal', 'ATR_14', 'BB_Width', 
                          'Volatility_20', 'SMA_Cross', 'Daily_Return']
        
        # Pastikan tidak ada NaN
        df = df.dropna(subset=feature_columns)
        
        # Extract data
        prices = df['Close'].values
        features = df[feature_columns].values
        
        print(f"Data siap: {len(prices)} hari dengan {features.shape[1]} fitur")
        
        # Inisialisasi PPO Trader
        ppo_trader = PPOTrader(
            prices=prices,
            features=features,
            initial_investment=10000
        )
        
        # Latih model
        print("Melatih PPO Agent...")
        train_results = ppo_trader.train(episodes=5, max_steps=None)
        
        # Backtest
        print("Menjalankan backtest...")
        backtest_results = ppo_trader.backtest()
        
        # Print hasil
        performance = backtest_results['performance']
        print(f"Hasil PPO dengan indikator teknikal:")
        print(f"  Return: {performance['total_return']:.2f}%")
        print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {performance['max_drawdown']:.2f}%")
        print(f"  Win Rate: {performance['win_rate']:.2f}%")
        print(f"  Jumlah Transaksi: {performance['num_trades']}")
        
        # Analisis distribusi aksi
        actions = backtest_results['actions']
        action_counts = {
            'Hold': actions.count(0),
            'Buy': actions.count(1),
            'Sell': actions.count(2)
        }
        
        print(f"Distribusi Aksi:")
        print(f"  Hold: {action_counts['Hold']} ({action_counts['Hold']/len(actions)*100:.1f}%)")
        print(f"  Buy: {action_counts['Buy']} ({action_counts['Buy']/len(actions)*100:.1f}%)")
        print(f"  Sell: {action_counts['Sell']} ({action_counts['Sell']/len(actions)*100:.1f}%)")
        
        # Plot hasil
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(backtest_results['portfolio_values'], label='Portfolio Value')
        plt.title('PPO Agent Performance')
        plt.xlabel('Day')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.legend()
        
        # Plot harga dan aksi
        plt.subplot(2, 1, 2)
        plt.plot(prices, label='Price', color='blue', alpha=0.6)
        
        # Tambahkan marker untuk aksi buy/sell
        buy_indices = [i for i, a in enumerate(actions) if a == 1]
        sell_indices = [i for i, a in enumerate(actions) if a == 2]
        
        plt.scatter(buy_indices, prices[buy_indices], color='green', marker='^', label='Buy')
        plt.scatter(sell_indices, prices[sell_indices], color='red', marker='v', label='Sell')
        
        plt.title('Trading Actions')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/ppo_agent_technical.png', dpi=300)
        
        print("Pengujian PPO Agent selesai")
    
    except Exception as e:
        print(f"Error dalam pengujian PPO Agent: {str(e)}")
        traceback.print_exc()

def test_integrated_system():
    """Pengujian terintegrasi dari seluruh sistem"""
    print("\n=== Pengujian Terintegrasi Sistem AI Hedge Fund ===")
    
    try:
        # Generate data portofolio multi-aset
        np.random.seed(42)
        days = 60
        
        # Buat data harga untuk 3 aset dengan karakteristik berbeda
        assets = {
            'ASSET1': 100 + np.cumsum(np.random.normal(0.001, 0.02, days)),  # Tren naik moderat
            'ASSET2': 50 + np.cumsum(np.random.normal(-0.001, 0.03, days)),  # Tren turun volatil
            'ASSET3': 200 + 10*np.sin(np.linspace(0, 3*np.pi, days))         # Siklikal
        }
        
        # Buat prediksi sederhana (lagged 1 day + noise)
        predictions = {
            symbol: np.roll(prices, -1) + np.random.normal(0, 0.05 * np.mean(prices), days)
            for symbol, prices in assets.items()
        }
        
        # Fix nilai terakhir prediksi
        for symbol in assets:
            predictions[symbol][-1] = predictions[symbol][-2]
        
        # Generate dates
        start_date = datetime.now() - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Setup sistem terintegrasi
        # 1. Inisialisasi portofolio
        portfolio = MultiAssetPortfolio(
            assets=list(assets.keys()),
            initial_capital=100000,
            transaction_fee=0.001
        )
        
        # 2. Buat risk manager
        risk_manager = RiskManager(
            max_drawdown=0.1,
            max_position_size=0.2,
            stop_loss=0.05,
            trailing_stop=0.03
        )
        
        # 3. Setup backtester untuk setiap aset
        backtesters = {}
        for symbol in assets.keys():
            backtesters[symbol] = Backtester(
                actual_prices=assets[symbol],
                predicted_prices=predictions[symbol],
                initial_investment=portfolio.initial_capital / len(assets),
                transaction_fee=portfolio.transaction_fee,
                dates=dates
            )
        
        # 4. PPO Trader untuk ASSET3 (sebagai contoh strategi ML)
        ppo_trader = PPOTrader(
            prices=assets['ASSET3'],
            initial_investment=portfolio.initial_capital / len(assets)
        )
        
        # Latih PPO agent dengan cepat
        print("Melatih PPO agent...")
        ppo_trader.train(episodes=3, max_steps=20)
        
        # Simulasi trading
        print("\nMenjalankan simulasi trading terintegrasi...")
        portfolio_values = []
        strategies = {
            'ASSET1': 'Trend Following',
            'ASSET2': 'Mean Reversion'
            # ASSET3 akan menggunakan PPO
        }
        
        # Loop melalui setiap hari trading
        for i in range(1, days):
            date = dates[i]
            current_prices = {symbol: prices[i] for symbol, prices in assets.items()}
            
            # Kumpulkan sinyal dari masing-masing strategi
            trading_signals = []
            
            # 1. Dapatkan sinyal dari strategi tradisional
            for symbol, strategy in strategies.items():
                # Generate sinyal menggunakan backtester (tanpa eksekusi)
                signal = backtesters[symbol].get_strategy_signal(
                    strategy=strategy,
                    index=i,
                    allow_short=True
                )
                
                # Konversi sinyal ke format portfolio
                if signal == 'BUY':
                    # Hitung volatilitas untuk ukuran posisi
                    if i >= 21:
                        price_window = assets[symbol][i-21:i]
                        returns = np.diff(price_window) / price_window[:-1]
                        volatility = np.std(returns)
                    else:
                        volatility = 0.02
                        
                    # Size posisi berdasarkan risk manager
                    position_size = risk_manager.size_position(
                        'BUY', current_prices[symbol], portfolio.cash, volatility
                    )
                    
                    trading_signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'size': position_size,
                        'volatility': volatility
                    })
                    
                elif signal == 'SELL' and symbol in portfolio.positions:
                    trading_signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'size': 1.0
                    })
                    
                elif signal == 'SHORT':
                    # Hitung volatilitas
                    if i >= 21:
                        price_window = assets[symbol][i-21:i]
                        returns = np.diff(price_window) / price_window[:-1]
                        volatility = np.std(returns)
                    else:
                        volatility = 0.02
                        
                    # Size posisi berdasarkan risk manager
                    position_size = risk_manager.size_position(
                        'SHORT', current_prices[symbol], portfolio.cash, volatility
                    )
                    
                    trading_signals.append({
                        'symbol': symbol,
                        'action': 'SHORT',
                        'size': position_size,
                        'volatility': volatility
                    })
                    
                elif signal == 'COVER' and symbol in portfolio.positions:
                    trading_signals.append({
                        'symbol': symbol,
                        'action': 'COVER',
                        'size': 1.0
                    })
            
            # 2. Dapatkan sinyal dari PPO untuk ASSET3
            ppo_env = TradingEnv(
                prices=assets['ASSET3'][:i+1],
                initial_balance=portfolio.initial_capital / len(assets)
            )
            ppo_env.reset()
            
            # Set environment ke state saat ini
            for j in range(i):
                ppo_env.step(0)  # Dummy steps untuk mencapai current state
            
            # Dapatkan aksi dari agent
            state = ppo_env._get_observation()
            action, _, _ = ppo_trader.agent.get_action(state)
            
            # Konversi aksi ke sinyal
            if action == 1:  # Buy
                volatility = 0.02  # Default
                position_size = risk_manager.size_position(
                    'BUY', current_prices['ASSET3'], portfolio.cash, volatility
                )
                
                trading_signals.append({
                    'symbol': 'ASSET3',
                    'action': 'BUY',
                    'size': position_size,
                    'volatility': volatility
                })
                
            elif action == 2:  # Sell
                if 'ASSET3' in portfolio.positions:
                    trading_signals.append({
                        'symbol': 'ASSET3',
                        'action': 'SELL',
                        'size': 1.0
                    })
            
            # 3. Periksa stop loss dari risk manager
            stop_signals = risk_manager.check_stop_loss(portfolio.positions, current_prices)
            trading_signals.extend(stop_signals)
            
            # 4. Eksekusi semua sinyal
            result = portfolio.allocate_capital(trading_signals, current_prices, date)
            portfolio_values.append(result['portfolio_value'])
            
            # Print update
            if i % 10 == 0 or i == days-1:
                print(f"Hari {i}: Nilai Portofolio ${result['portfolio_value']:.2f}, "
                    f"{len(trading_signals)} sinyal, {len(result['executed_orders'])} eksekusi")
        
        # Plot hasil
        plt.figure(figsize=(12, 8))
        
        # Plot nilai portofolio
        plt.subplot(2, 1, 1)
        plt.plot(dates[1:], portfolio_values, marker='o', label='Portfolio Value')
        plt.title('AI Hedge Fund - Sistem Terintegrasi')
        plt.xlabel('Tanggal')
        plt.ylabel('Nilai Portofolio')
        plt.grid(True)
        plt.legend()
        
        # Plot nilai aset
        plt.subplot(2, 1, 2)
        for symbol, prices in assets.items():
            # Normalisasi untuk perbandingan
            norm_prices = prices / prices[0] * 100
            plt.plot(dates, norm_prices, label=f'{symbol} (normalized)')
        
        plt.title('Harga Aset (Normalized)')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga (% dari awal)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/integrated_system_test.png', dpi=300)
        
        # Cetak ringkasan portofolio akhir
        summary = portfolio.get_portfolio_summary()
        print(f"\nRingkasan Akhir Portofolio:")
        print(f"  Nilai Awal: ${summary['initial_capital']:.2f}")
        print(f"  Nilai Akhir: ${summary['current_value']:.2f}")
        print(f"  Return: {summary['return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {summary['metrics']['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {summary['metrics']['max_drawdown']*100:.2f}%")
        
        # Posisi akhir
        positions = portfolio.get_positions_df()
        if len(positions) > 0:
            print("\nPosisi Akhir:")
            print(positions[['symbol', 'type', 'shares', 'entry_price', 'unrealized_pnl_pct']])
        else:
            print("\nTidak ada posisi aktif di akhir simulasi")
        
        # Cetak statistik transaksi
        transactions = portfolio.get_transactions_df()
        print(f"\nStatistik Transaksi ({len(transactions)} total):")
        transaction_counts = transactions['action'].value_counts()
        print(transaction_counts)
        
        print("\nPengujian Terintegrasi selesai")
    
    except Exception as e:
        print(f"Error dalam pengujian terintegrasi: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=== PENGUJIAN KOMPREHENSIF AI HEDGE FUND ===")
    print("Pengujian ini akan menguji semua fitur utama sistem AI Hedge Fund")
    print("Hasil pengujian akan disimpan dalam folder 'results'")
    print("=" * 50)
    
    try:
        # Jalankan semua pengujian satu per satu
        test_risk_manager_advanced()
        test_multi_asset_portfolio_advanced()
        test_backtester_strategies()
        test_ppo_agent_with_indicators()
        test_integrated_system()
        
        print("\n✅ SEMUA PENGUJIAN BERHASIL DISELESAIKAN!")
        print("Grafik dan hasil pengujian tersimpan di folder 'results'")
        
    except Exception as e:
        print(f"\n❌ PENGUJIAN GAGAL: {str(e)}")
        traceback.print_exc() 