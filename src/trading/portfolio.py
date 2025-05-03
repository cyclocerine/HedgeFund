"""
Portfolio Management Module
========================

Modul ini berisi implementasi kelas MultiAssetPortfolio
untuk mengelola portofolio multi-aset.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from .risk_manager import RiskManager

class MultiAssetPortfolio:
    def __init__(self, assets, initial_capital=100000, transaction_fee=0.001):
        """
        Inisialisasi Multi-Asset Portfolio
        
        Parameters:
        -----------
        assets : list
            List simbol aset yang akan diperdagangkan
        initial_capital : float
            Modal awal
        transaction_fee : float
            Biaya transaksi sebagai persentase dari nilai transaksi
        """
        self.assets = assets  # List simbol aset
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # {symbol: {'shares': x, 'entry_price': y, 'type': 'LONG/SHORT'}}
        self.cash = initial_capital
        self.transaction_history = []
        self.portfolio_history = []
        self.transaction_fee = transaction_fee
        self.risk_manager = RiskManager()
        
    def update_prices(self, current_prices, timestamp=None):
        """
        Update nilai portofolio berdasarkan harga terbaru
        
        Parameters:
        -----------
        current_prices : dict
            Dictionary harga saat ini {symbol: price}
        timestamp : datetime, optional
            Timestamp untuk update ini
            
        Returns:
        --------
        float
            Nilai portofolio saat ini
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        portfolio_value = self.cash
        
        # Update nilai setiap posisi
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                if position['type'] == 'LONG':
                    position_value = position['shares'] * current_price
                    portfolio_value += position_value
                    
                    # Update P&L
                    position['current_value'] = position_value
                    position['unrealized_pnl'] = position_value - (position['shares'] * position['entry_price'])
                    position['unrealized_pnl_pct'] = (current_price / position['entry_price'] - 1) * 100
                    
                elif position['type'] == 'SHORT':
                    # Untuk short, nilai posisi adalah negatif
                    position_value = position['shares'] * current_price
                    portfolio_value -= position_value  # Kurangi dari portofolio
                    
                    # Update P&L
                    position['current_value'] = position_value
                    position['unrealized_pnl'] = (position['shares'] * position['entry_price']) - position_value
                    position['unrealized_pnl_pct'] = (position['entry_price'] / current_price - 1) * 100
        
        # Catat history portofolio
        self.portfolio_history.append({
            'timestamp': timestamp,
            'value': portfolio_value,
            'cash': self.cash
        })
        
        return portfolio_value
        
    def allocate_capital(self, signals, current_prices, timestamp=None):
        """
        Alokasikan modal berdasarkan sinyal trading
        
        Parameters:
        -----------
        signals : list
            List sinyal trading [{symbol, action, size, volatility}]
        current_prices : dict
            Dictionary harga saat ini {symbol: price}
        timestamp : datetime, optional
            Timestamp untuk transaksi ini
            
        Returns:
        --------
        dict
            Hasil alokasi modal
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        portfolio_value = self.update_prices(current_prices, timestamp)
        executed_orders = []
        
        # Check portfolio risk first
        risk_action = self.risk_manager.check_portfolio_risk(portfolio_value, self.positions)
        
        # Reduce exposure if needed
        if risk_action['reduce_exposure']:
            self._reduce_exposure(risk_action['target_exposure'], current_prices, timestamp)
        
        # Check stop losses
        stop_loss_signals = self.risk_manager.check_stop_loss(self.positions, current_prices)
        signals = stop_loss_signals + signals  # Prioritas sinyal stop loss
        
        # Process signals
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            
            # Skip if price not available
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Default volatility jika tidak ada
            volatility = signal.get('volatility', 0.02)
            
            # Ukuran posisi dari sinyal atau hitung otomatis
            if 'size' in signal:
                position_size = signal['size']
            else:
                position_size = self.risk_manager.size_position(
                    action, current_price, portfolio_value, volatility
                )
            
            # Execute order
            if action == 'BUY':
                order_result = self._execute_buy(symbol, current_price, position_size, portfolio_value, timestamp)
                if order_result:
                    executed_orders.append(order_result)
            elif action == 'SELL':
                order_result = self._execute_sell(symbol, current_price, position_size, timestamp)
                if order_result:
                    executed_orders.append(order_result)
            elif action == 'SHORT':
                order_result = self._execute_short(symbol, current_price, position_size, portfolio_value, timestamp)
                if order_result:
                    executed_orders.append(order_result)
            elif action == 'COVER':
                order_result = self._execute_cover(symbol, current_price, position_size, timestamp)
                if order_result:
                    executed_orders.append(order_result)
        
        return {
            'portfolio_value': portfolio_value,
            'executed_orders': executed_orders
        }
    
    def _execute_buy(self, symbol, price, position_size, portfolio_value, timestamp=None):
        """
        Eksekusi pembelian aset
        
        Parameters:
        -----------
        symbol : str
            Simbol aset
        price : float
            Harga aset
        position_size : float
            Ukuran posisi sebagai persentase dari portofolio
        portfolio_value : float
            Nilai portofolio saat ini
        timestamp : datetime, optional
            Timestamp untuk transaksi ini
            
        Returns:
        --------
        dict
            Hasil eksekusi order
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Hitung jumlah yang akan digunakan
        available_cash = min(self.cash, portfolio_value * position_size)
        
        if available_cash <= 0:
            return None
            
        # Hitung jumlah saham yang dapat dibeli
        shares_to_buy = available_cash / (price * (1 + self.transaction_fee))
        cost = shares_to_buy * price
        fee = cost * self.transaction_fee
        
        # Update cash
        self.cash -= (cost + fee)
        
        # Jika sudah memiliki posisi long, tambahkan ke posisi tersebut
        if symbol in self.positions and self.positions[symbol]['type'] == 'LONG':
            # Hitung harga rata-rata baru
            total_shares = self.positions[symbol]['shares'] + shares_to_buy
            avg_price = (self.positions[symbol]['shares'] * self.positions[symbol]['entry_price'] + cost) / total_shares
            
            self.positions[symbol]['shares'] = total_shares
            self.positions[symbol]['entry_price'] = avg_price
        else:
            # Buat posisi baru
            self.positions[symbol] = {
                'shares': shares_to_buy,
                'entry_price': price,
                'type': 'LONG',
                'entry_date': timestamp,
                'high_price': price  # Untuk trailing stop
            }
        
        # Catat transaksi
        transaction = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'shares': shares_to_buy,
            'value': cost,
            'fee': fee,
            'position_size': position_size
        }
        
        self.transaction_history.append(transaction)
        
        return transaction
    
    def _execute_sell(self, symbol, price, position_size=1.0, timestamp=None):
        """
        Eksekusi penjualan aset
        
        Parameters:
        -----------
        symbol : str
            Simbol aset
        price : float
            Harga aset
        position_size : float
            Persentase posisi yang akan dijual (1.0 = 100%)
        timestamp : datetime, optional
            Timestamp untuk transaksi ini
            
        Returns:
        --------
        dict
            Hasil eksekusi order
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Periksa apakah memiliki posisi
        if symbol not in self.positions or self.positions[symbol]['type'] != 'LONG':
            return None
            
        # Hitung jumlah saham yang akan dijual
        shares_to_sell = self.positions[symbol]['shares'] * position_size
        value = shares_to_sell * price
        fee = value * self.transaction_fee
        
        # Hitung realized P&L
        entry_value = shares_to_sell * self.positions[symbol]['entry_price']
        realized_pnl = value - entry_value - fee
        realized_pnl_pct = (price / self.positions[symbol]['entry_price'] - 1) * 100 - (self.transaction_fee * 100)
        
        # Update cash
        self.cash += (value - fee)
        
        # Update posisi
        if position_size >= 1.0:
            del self.positions[symbol]
        else:
            self.positions[symbol]['shares'] -= shares_to_sell
        
        # Catat transaksi
        transaction = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'SELL',
            'price': price,
            'shares': shares_to_sell,
            'value': value,
            'fee': fee,
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': realized_pnl_pct,
            'position_size': position_size
        }
        
        self.transaction_history.append(transaction)
        
        return transaction
    
    def _execute_short(self, symbol, price, position_size, portfolio_value, timestamp=None):
        """
        Eksekusi short selling aset
        
        Parameters:
        -----------
        symbol : str
            Simbol aset
        price : float
            Harga aset
        position_size : float
            Ukuran posisi sebagai persentase dari portofolio
        portfolio_value : float
            Nilai portofolio saat ini
        timestamp : datetime, optional
            Timestamp untuk transaksi ini
            
        Returns:
        --------
        dict
            Hasil eksekusi order
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Hitung jumlah yang akan digunakan
        short_value = portfolio_value * position_size
        
        # Hitung jumlah saham yang dapat di-short
        shares_to_short = short_value / price
        fee = short_value * self.transaction_fee
        
        # Untuk short, kita mendapatkan uang dari pinjaman saham
        self.cash += (short_value - fee)
        
        # Jika sudah memiliki posisi short, tambahkan ke posisi tersebut
        if symbol in self.positions and self.positions[symbol]['type'] == 'SHORT':
            # Hitung harga rata-rata baru
            total_shares = self.positions[symbol]['shares'] + shares_to_short
            avg_price = (self.positions[symbol]['shares'] * self.positions[symbol]['entry_price'] + short_value) / total_shares
            
            self.positions[symbol]['shares'] = total_shares
            self.positions[symbol]['entry_price'] = avg_price
        else:
            # Buat posisi baru
            self.positions[symbol] = {
                'shares': shares_to_short,
                'entry_price': price,
                'type': 'SHORT',
                'entry_date': timestamp,
                'low_price': price  # Untuk trailing stop
            }
        
        # Catat transaksi
        transaction = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'SHORT',
            'price': price,
            'shares': shares_to_short,
            'value': short_value,
            'fee': fee,
            'position_size': position_size
        }
        
        self.transaction_history.append(transaction)
        
        return transaction
    
    def _execute_cover(self, symbol, price, position_size=1.0, timestamp=None):
        """
        Tutup posisi short
        
        Parameters:
        -----------
        symbol : str
            Simbol aset
        price : float
            Harga aset
        position_size : float
            Persentase posisi yang akan ditutup (1.0 = 100%)
        timestamp : datetime, optional
            Timestamp untuk transaksi ini
            
        Returns:
        --------
        dict
            Hasil eksekusi order
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Periksa apakah memiliki posisi short
        if symbol not in self.positions or self.positions[symbol]['type'] != 'SHORT':
            return None
            
        # Hitung jumlah saham yang akan ditutup
        shares_to_cover = self.positions[symbol]['shares'] * position_size
        cover_cost = shares_to_cover * price
        fee = cover_cost * self.transaction_fee
        
        # Hitung realized P&L
        entry_value = shares_to_cover * self.positions[symbol]['entry_price']
        realized_pnl = entry_value - cover_cost - fee
        realized_pnl_pct = (self.positions[symbol]['entry_price'] / price - 1) * 100 - (self.transaction_fee * 100)
        
        # Update cash
        self.cash -= (cover_cost + fee)
        
        # Update posisi
        if position_size >= 1.0:
            del self.positions[symbol]
        else:
            self.positions[symbol]['shares'] -= shares_to_cover
        
        # Catat transaksi
        transaction = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'COVER',
            'price': price,
            'shares': shares_to_cover,
            'value': cover_cost,
            'fee': fee,
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': realized_pnl_pct,
            'position_size': position_size
        }
        
        self.transaction_history.append(transaction)
        
        return transaction
    
    def _reduce_exposure(self, target_exposure, current_prices, timestamp=None):
        """
        Kurangi eksposur portofolio ke target tertentu
        
        Parameters:
        -----------
        target_exposure : float
            Eksposur target sebagai persentase dari portofolio
        current_prices : dict
            Dictionary harga saat ini {symbol: price}
        timestamp : datetime, optional
            Timestamp untuk transaksi ini
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        portfolio_value = self.update_prices(current_prices, timestamp)
        current_exposure = sum(pos['current_value'] for pos in self.positions.values() if pos['type'] == 'LONG')
        current_exposure_ratio = current_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Jika eksposur saat ini lebih tinggi dari target, kurangi posisi
        if current_exposure_ratio > target_exposure:
            reduction_ratio = 1 - (target_exposure / current_exposure_ratio)
            
            # Kurangi setiap posisi secara proporsional
            for symbol, position in list(self.positions.items()):
                if position['type'] == 'LONG' and symbol in current_prices:
                    self._execute_sell(symbol, current_prices[symbol], reduction_ratio, timestamp)
                elif position['type'] == 'SHORT' and symbol in current_prices:
                    self._execute_cover(symbol, current_prices[symbol], reduction_ratio, timestamp)
    
    def get_portfolio_summary(self):
        """
        Dapatkan ringkasan portofolio
        
        Returns:
        --------
        dict
            Ringkasan portofolio
        """
        if len(self.portfolio_history) == 0:
            return {
                'initial_capital': self.initial_capital,
                'current_value': self.cash,
                'cash': self.cash,
                'positions': 0,
                'return_pct': 0,
                'metrics': {
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'volatility': 0,
                    'var_95': 0
                }
            }
        
        current_value = self.portfolio_history[-1]['value']
        
        # Hitung return
        return_pct = (current_value / self.initial_capital - 1) * 100
        
        # Hitung metrik portofolio
        portfolio_values = [p['value'] for p in self.portfolio_history]
        metrics = self.risk_manager.calculate_portfolio_metrics(portfolio_values)
        
        return {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'cash': self.cash,
            'positions': len(self.positions),
            'return_pct': return_pct,
            'metrics': metrics
        }
    
    def get_transactions_df(self):
        """
        Dapatkan history transaksi sebagai DataFrame
        
        Returns:
        --------
        pandas.DataFrame
            History transaksi
        """
        return pd.DataFrame(self.transaction_history)
    
    def get_positions_df(self):
        """
        Dapatkan posisi aktif sebagai DataFrame
        
        Returns:
        --------
        pandas.DataFrame
            Posisi aktif
        """
        positions_list = []
        
        for symbol, position in self.positions.items():
            pos_dict = position.copy()
            pos_dict['symbol'] = symbol
            positions_list.append(pos_dict)
            
        if not positions_list:
            return pd.DataFrame()
            
        return pd.DataFrame(positions_list)
    
    def get_portfolio_history_df(self):
        """
        Dapatkan history portofolio sebagai DataFrame
        
        Returns:
        --------
        pandas.DataFrame
            History portofolio
        """
        return pd.DataFrame(self.portfolio_history) 