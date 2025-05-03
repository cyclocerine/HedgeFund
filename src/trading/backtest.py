"""
Backtesting Module
================

Modul ini berisi implementasi kelas Backtester untuk
menguji strategi trading pada data historis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .strategies import TradingStrategy
from .risk_manager import RiskManager
from .portfolio import MultiAssetPortfolio

class Backtester:
    def __init__(self, actual_prices, predicted_prices, initial_investment=10000, transaction_fee=0.001, 
                 dates=None, features=None):
        """
        Inisialisasi Backtester
        
        Parameters:
        -----------
        actual_prices : array-like
            Array harga aktual historis
        predicted_prices : array-like
            Array harga prediksi dari model
        initial_investment : float, optional
            Jumlah investasi awal, default 10000
        transaction_fee : float, optional
            Biaya transaksi sebagai persentase dari nilai transaksi, default 0.001 (0.1%)
        dates : array-like, optional
            Array tanggal yang sesuai dengan data harga (untuk analisis lebih lanjut)
        features : array-like, optional
            Fitur tambahan yang digunakan untuk strategi trading
        """
        self.actual_prices = actual_prices
        self.predicted_prices = predicted_prices
        self.initial_investment = initial_investment
        self.transaction_fee = transaction_fee
        self.dates = dates
        self.features = features
        
        # Inisialisasi portfolio dan risk manager
        self.portfolio = None
        self.risk_manager = RiskManager()
        
    def run(self, strategy, params=None, allow_short=False, max_position_size=1.0):
        """
        Menjalankan backtesting untuk strategi tertentu
        
        Parameters:
        -----------
        strategy : str
            Nama strategi trading yang akan digunakan
        params : dict, optional
            Parameter tambahan untuk strategi
        allow_short : bool, optional
            Izinkan short selling, default False
        max_position_size : float, optional
            Ukuran posisi maksimum sebagai persentase dari portofolio (0.0-1.0)
            
        Returns:
        --------
        tuple
            (portfolio_values, trades, metrics_performance)
            - portfolio_values: nilai portfolio per hari
            - trades: list transaksi yang dilakukan
            - metrics_performance: metrik performa strategi
        """
        # Inisialisasi portfolio
        self.portfolio = MultiAssetPortfolio(
            assets=['SYMBOL'],  # Tanda pengganti untuk satu aset
            initial_capital=self.initial_investment,
            transaction_fee=self.transaction_fee
        )
        
        # Memastikan data memiliki panjang yang sama
        length = min(len(self.actual_prices), len(self.predicted_prices))
        actual_prices = self.actual_prices[:length]
        predicted_prices = self.predicted_prices[:length]
        
        # Buat dates jika tidak disediakan
        if self.dates is None:
            base_date = datetime.now() - timedelta(days=length)
            self.dates = [base_date + timedelta(days=i) for i in range(length)]
        else:
            self.dates = self.dates[:length]
            
        # Dapatkan fungsi strategi
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        
        # Iterasi melalui harga historis
        for i in range(1, length):
            date = self.dates[i] if i < len(self.dates) else datetime.now()
            current_price = actual_prices[i]
            
            # Siapkan harga saat ini sebagai dictionary untuk portfolio
            current_prices = {'SYMBOL': current_price}
            
            # Dapatkan sinyal dari strategi
            signal = strategy_function(predicted_prices, actual_prices, i, params)
            
            # Konversi signal dari format lama ('BUY', 'SELL', 'HOLD') ke format baru
            trading_signals = []
            
            # Hitung volatilitas untuk ukuran posisi
            if i >= 21:  # Minimal 21 hari untuk volatilitas
                # Perbaikan disini - pastikan array yang dibagi memiliki ukuran yang sama
                price_window = actual_prices[i-21:i]
                returns = np.diff(price_window) / price_window[:-1]
                volatility = np.std(returns)
            else:
                volatility = 0.02  # Default jika tidak cukup data
                
            # Penanganan sinyal
            if signal == 'BUY':
                # Batas ukuran posisi dari parameter
                position_size = min(max_position_size, 
                                  self.risk_manager.size_position('BUY', current_price, 
                                                                 self.portfolio.cash, volatility))
                
                trading_signals.append({
                    'symbol': 'SYMBOL',
                    'action': 'BUY',
                    'size': position_size,
                    'volatility': volatility
                })
                
            elif signal == 'SELL':
                trading_signals.append({
                    'symbol': 'SYMBOL',
                    'action': 'SELL',
                    'size': 1.0,  # Jual semua
                    'volatility': volatility
                })
                
            elif signal == 'SHORT' and allow_short:
                # Batas ukuran posisi dari parameter
                position_size = min(max_position_size, 
                                  self.risk_manager.size_position('SHORT', current_price, 
                                                                 self.portfolio.cash, volatility))
                
                trading_signals.append({
                    'symbol': 'SYMBOL',
                    'action': 'SHORT',
                    'size': position_size,
                    'volatility': volatility
                })
                
            elif signal == 'COVER' and allow_short:
                trading_signals.append({
                    'symbol': 'SYMBOL',
                    'action': 'COVER',
                    'size': 1.0,  # Tutup semua posisi short
                    'volatility': volatility
                })
            
            # Jalankan alokasi kapital
            self.portfolio.allocate_capital(trading_signals, current_prices, timestamp=date)
        
        # Dapatkan nilai history portofolio
        portfolio_history_df = self.portfolio.get_portfolio_history_df()
        portfolio_values = portfolio_history_df['value'].tolist()
        
        # Dapatkan transaksi
        transactions_df = self.portfolio.get_transactions_df()
        
        # Jika tidak ada transaksi, buat df kosong 
        if len(transactions_df) == 0:
            trades = []
        else:
            # Konversi DataFrame ke format lama untuk kompatibilitas
            trades = []
            for _, row in transactions_df.iterrows():
                trades.append({
                    'day': self.dates.index(row['timestamp']) if row['timestamp'] in self.dates else 0,
                    'type': row['action'],
                    'price': row['price'],
                    'shares': row['shares'],
                    'value': row['value']
                })
        
        # Dapatkan metrics terakhir
        metrics = self.portfolio.get_portfolio_summary()
        
        # Isi metrics sesuai format lama untuk kompatibilitas
        performance = {
            'initial_investment': self.initial_investment,
            'final_value': metrics['current_value'],
            'total_return': metrics['return_pct'],
            'max_drawdown': metrics['metrics']['max_drawdown'] * 100,
            'sharpe_ratio': metrics['metrics']['sharpe_ratio'],
            'win_rate': self._calculate_win_rate(transactions_df),
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance
    
    def run_multi_asset(self, strategies, assets_data, params=None):
        """
        Menjalankan backtesting untuk multiple aset
        
        Parameters:
        -----------
        strategies : dict
            Dictionary strategi per aset {'SYMBOL': 'strategy_name'}
        assets_data : dict
            Dictionary data per aset {'SYMBOL': {'actual': [...], 'predicted': [...], 'dates': [...]}}
        params : dict, optional
            Parameter tambahan {'SYMBOL': {...}}
            
        Returns:
        --------
        tuple
            (portfolio_df, transactions_df, metrics)
        """
        # Inisialisasi portfolio
        self.portfolio = MultiAssetPortfolio(
            assets=list(assets_data.keys()),
            initial_capital=self.initial_investment,
            transaction_fee=self.transaction_fee
        )
        
        # Dapatkan tanggal universal dari semua aset
        all_dates = set()
        for symbol, data in assets_data.items():
            if 'dates' in data:
                all_dates.update(data['dates'])
                
        all_dates = sorted(list(all_dates))
        
        # Iterasi melalui setiap tanggal
        for date in all_dates:
            trading_signals = []
            current_prices = {}
            
            # Periksa setiap aset
            for symbol, data in assets_data.items():
                # Skip jika tidak ada data untuk tanggal ini
                if 'dates' not in data or date not in data['dates']:
                    continue
                    
                # Dapatkan indeks untuk tanggal ini
                date_index = data['dates'].index(date)
                actual_prices = data['actual']
                predicted_prices = data['predicted']
                
                # Skip jika tidak cukup data
                if date_index < 1 or date_index >= len(actual_prices):
                    continue
                    
                # Set harga saat ini untuk portfolio
                current_prices[symbol] = actual_prices[date_index]
                
                # Dapatkan fungsi strategi
                strategy_name = strategies.get(symbol, 'Predictive')  # Default ke Predictive
                strategy_function = TradingStrategy.get_strategy_function(strategy_name)
                
                # Dapatkan parameter untuk aset ini
                asset_params = params.get(symbol, {}) if params else {}
                
                # Dapatkan sinyal
                signal = strategy_function(predicted_prices, actual_prices, date_index, asset_params)
                
                # Hitung volatilitas
                if date_index >= 21:
                    returns = np.diff(actual_prices[date_index-21:date_index]) / actual_prices[date_index-21:-1]
                    volatility = np.std(returns)
                else:
                    volatility = 0.02
                
                # Konversi sinyal
                if signal == 'BUY':
                    trading_signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'volatility': volatility
                    })
                elif signal == 'SELL':
                    trading_signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'volatility': volatility
                    })
                elif signal == 'SHORT':
                    trading_signals.append({
                        'symbol': symbol,
                        'action': 'SHORT',
                        'volatility': volatility
                    })
                elif signal == 'COVER':
                    trading_signals.append({
                        'symbol': symbol,
                        'action': 'COVER',
                        'volatility': volatility
                    })
            
            # Update portfolio jika ada harga
            if current_prices:
                self.portfolio.allocate_capital(trading_signals, current_prices, timestamp=date)
        
        # Dapatkan hasil
        portfolio_df = self.portfolio.get_portfolio_history_df()
        transactions_df = self.portfolio.get_transactions_df()
        metrics = self.portfolio.get_portfolio_summary()
        
        return portfolio_df, transactions_df, metrics
    
    def _calculate_win_rate(self, transactions_df):
        """
        Hitung win rate dari transaction dataframe
        
        Parameters:
        -----------
        transactions_df : pandas.DataFrame
            DataFrame transaksi
            
        Returns:
        --------
        float
            Win rate sebagai persentase
        """
        if len(transactions_df) == 0:
            return 0
            
        # Jika ada kolom realized_pnl, gunakan itu
        if 'realized_pnl' in transactions_df.columns:
            win_trades = transactions_df[transactions_df['realized_pnl'] > 0].shape[0]
            total_closed_trades = transactions_df[transactions_df['action'].isin(['SELL', 'COVER'])].shape[0]
            
            return (win_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
        
        # Jika tidak, hitung secara manual (kompatibilitas)
        win_trades = 0
        loss_trades = 0
        
        # Kelompokkan transaksi beli/jual
        buy_sells = []
        current_buy = None
        
        for _, row in transactions_df.iterrows():
            if row['action'] == 'BUY':
                current_buy = row
            elif row['action'] == 'SELL' and current_buy is not None:
                buy_sells.append((current_buy, row))
                current_buy = None
            elif row['action'] == 'SHORT':
                current_buy = row
            elif row['action'] == 'COVER' and current_buy is not None:
                buy_sells.append((current_buy, row))
                current_buy = None
        
        # Hitung profit/loss untuk setiap pasangan
        for buy, sell in buy_sells:
            if buy['action'] == 'BUY':
                profit = (sell['price'] - buy['price']) / buy['price']
            else:  # SHORT
                profit = (buy['price'] - sell['price']) / buy['price']
                
            if profit > 0:
                win_trades += 1
            else:
                loss_trades += 1
        
        total_trades = win_trades + loss_trades
        return (win_trades / total_trades * 100) if total_trades > 0 else 0
            
    def plot_results(self, portfolio_values=None, benchmark=None):
        """
        Plot hasil backtest
        
        Parameters:
        -----------
        portfolio_values : array-like, optional
            Nilai portofolio per hari, default menggunakan history dari portfolio object
        benchmark : array-like, optional
            Array harga benchmark untuk perbandingan
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object untuk plotting lebih lanjut
        """
        import matplotlib.pyplot as plt
        
        if portfolio_values is None and self.portfolio is not None:
            portfolio_df = self.portfolio.get_portfolio_history_df()
            portfolio_values = portfolio_df['value'].values
            dates = portfolio_df['timestamp'].values
        else:
            # Buat tanggal dummy jika tidak ada
            dates = self.dates[:len(portfolio_values)] if self.dates is not None else range(len(portfolio_values))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot portofolio
        ax.plot(dates, portfolio_values, label='Portfolio', linewidth=2)
        
        # Plot benchmark jika ada
        if benchmark is not None:
            # Normalisasi benchmark ke nilai awal portofolio
            benchmark_norm = [b * (portfolio_values[0] / benchmark[0]) for b in benchmark]
            ax.plot(dates, benchmark_norm[:len(dates)], label='Benchmark', linewidth=2, linestyle='--')
        
        # Plot transaksi
        if self.portfolio is not None:
            transactions_df = self.portfolio.get_transactions_df()
            
            for _, row in transactions_df.iterrows():
                if row['action'] == 'BUY':
                    marker = '^'
                    color = 'green'
                elif row['action'] == 'SELL':
                    marker = 'v'
                    color = 'red'
                elif row['action'] == 'SHORT':
                    marker = 'v'
                    color = 'purple'
                elif row['action'] == 'COVER':
                    marker = '^'
                    color = 'orange'
                else:
                    continue
                
                # Cari nilai portfolio pada timestamp ini
                portfolio_idx = portfolio_df[portfolio_df['timestamp'] == row['timestamp']].index
                if len(portfolio_idx) > 0:
                    y_val = portfolio_df.iloc[portfolio_idx[0]]['value']
                    ax.scatter(row['timestamp'], y_val, marker=marker, color=color, s=100)
        
        # Format plot
        ax.set_title('Backtest Results', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax.legend()
        ax.grid(True)
        
        return fig 

    def get_strategy_signal(self, strategy, index, allow_short=False):
        """
        Mendapatkan sinyal strategi pada index tertentu tanpa menjalankan simulasi lengkap
        
        Parameters:
        -----------
        strategy : str
            Nama strategi trading yang akan digunakan
        index : int
            Indeks waktu untuk mendapatkan sinyal
        allow_short : bool, optional
            Izinkan sinyal short selling, default False
            
        Returns:
        --------
        str
            Sinyal trading ('BUY', 'SELL', 'SHORT', 'COVER', atau 'HOLD')
        """
        # Memastikan index valid
        if index < 1 or index >= len(self.actual_prices):
            return 'HOLD'
            
        # Dapatkan fungsi strategi
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        
        # Dapatkan sinyal dari strategi
        signal = strategy_function(self.predicted_prices, self.actual_prices, index, None)
        
        # Jika short tidak diizinkan, konversi sinyal SHORT/COVER ke HOLD
        if not allow_short and signal in ['SHORT', 'COVER']:
            return 'HOLD'
            
        return signal 