"""
Risk Manager Module
================

Modul ini berisi implementasi kelas RiskManager untuk
mengelola risiko trading secara otomatis.
"""

import numpy as np

class RiskManager:
    def __init__(self, max_drawdown=0.1, max_position_size=0.2, stop_loss=0.05, 
                 trailing_stop=0.03, max_capital_per_trade=0.2):
        """
        Inisialisasi Risk Manager
        
        Parameters:
        -----------
        max_drawdown : float
            Drawdown maksimum yang diizinkan sebelum mengurangi eksposur
        max_position_size : float
            Ukuran posisi maksimum sebagai persentase dari portofolio
        stop_loss : float
            Stop loss per posisi (sebagai persentase dari harga masuk)
        trailing_stop : float
            Trailing stop loss (sebagai persentase dari harga tertinggi/terendah)
        max_capital_per_trade : float
            Maksimum kapital yang digunakan per perdagangan
        """
        self.max_drawdown = max_drawdown             # Drawdown maksimum yang diizinkan
        self.max_position_size = max_position_size   # Ukuran posisi maksimum (% dari portofolio)
        self.stop_loss = stop_loss                   # Stop loss per posisi
        self.trailing_stop = trailing_stop           # Trailing stop loss
        self.max_capital_per_trade = max_capital_per_trade  # Max kapital per trade
        self.peak_value = 0                          # Nilai peak portofolio
    
    def check_portfolio_risk(self, portfolio_value, positions):
        """
        Periksa risiko portofolio secara keseluruhan
        
        Parameters:
        -----------
        portfolio_value : float
            Nilai portofolio saat ini
        positions : dict
            Posisi yang sedang aktif
            
        Returns:
        --------
        dict
            Keputusan manajemen risiko
        """
        # Update peak value
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # Hitung drawdown saat ini
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
        
        # Jika drawdown melebihi batas, kurangi eksposur
        if current_drawdown > self.max_drawdown:
            return {
                'reduce_exposure': True,
                'target_exposure': 0.5  # Kurangi eksposur ke 50%
            }
        
        return {'reduce_exposure': False}
    
    def size_position(self, signal, current_price, portfolio_value, volatility):
        """
        Tentukan ukuran posisi berdasarkan volatilitas dan risiko
        
        Parameters:
        -----------
        signal : str
            Sinyal trading (BUY, SELL, SHORT, COVER)
        current_price : float
            Harga aset saat ini
        portfolio_value : float
            Nilai portofolio saat ini
        volatility : float
            Volatilitas aset (contoh: standar deviasi return)
            
        Returns:
        --------
        float
            Ukuran posisi yang direkomendasikan (persentase dari portofolio)
        """
        # Batas ukuran trade berdasarkan kapital
        capital_limit = self.max_capital_per_trade
        
        # Kelly Criterion (disederhanakan): f* = (edge/odds)
        # Diasumsikan edge = 0.05 (5%) dan odds = volatility * 10
        edge = 0.05  # Asumsi edge default
        
        # Sesuaikan ukuran posisi berdasarkan volatilitas
        vol_adjusted_size = edge / (volatility * 10) if volatility > 0 else self.max_position_size
        
        # Batasi ukuran posisi
        position_size = min(vol_adjusted_size, self.max_position_size, capital_limit)
        
        return position_size
    
    def check_stop_loss(self, positions, current_prices):
        """
        Periksa apakah posisi perlu dilikuidasi berdasarkan stop loss
        
        Parameters:
        -----------
        positions : dict
            Dictionary posisi aktif {symbol: {data posisi}}
        current_prices : dict
            Dictionary harga saat ini {symbol: price}
            
        Returns:
        --------
        list
            List sinyal likuidasi
        """
        signals = []
        
        for symbol, pos in positions.items():
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            if pos['type'] == 'LONG':
                entry_price = pos['entry_price']
                
                # Update high price jika harga saat ini lebih tinggi
                if 'high_price' not in pos or current_price > pos['high_price']:
                    pos['high_price'] = current_price
                
                # Hitung kerugian saat ini
                loss = (entry_price - current_price) / entry_price
                
                # Hitung trailing stop
                trailing_stop_price = pos['high_price'] * (1 - self.trailing_stop)
                
                if loss > self.stop_loss or current_price < trailing_stop_price:
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'reason': 'STOP_LOSS' if loss > self.stop_loss else 'TRAILING_STOP'
                    })
            
            elif pos['type'] == 'SHORT':
                entry_price = pos['entry_price']
                
                # Update low price jika harga saat ini lebih rendah
                if 'low_price' not in pos or current_price < pos['low_price']:
                    pos['low_price'] = current_price
                
                # Hitung kerugian untuk posisi short
                loss = (current_price - entry_price) / entry_price
                
                # Hitung trailing stop
                trailing_stop_price = pos['low_price'] * (1 + self.trailing_stop)
                
                if loss > self.stop_loss or current_price > trailing_stop_price:
                    signals.append({
                        'symbol': symbol,
                        'action': 'COVER',
                        'reason': 'STOP_LOSS' if loss > self.stop_loss else 'TRAILING_STOP'
                    })
        
        return signals
    
    def calculate_portfolio_metrics(self, portfolio_history):
        """
        Hitung metrik risiko portofolio
        
        Parameters:
        -----------
        portfolio_history : list
            History nilai portofolio
            
        Returns:
        --------
        dict
            Metrik risiko portofolio
        """
        if len(portfolio_history) < 2:
            return {
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'var_95': 0
            }
            
        # Hitung return harian
        returns = np.diff(portfolio_history) / portfolio_history[:-1]
        
        # Volatilitas (standar deviasi return)
        volatility = np.std(returns)
        
        # Sharpe ratio (asumsi risk-free rate = 0)
        sharpe_ratio = np.mean(returns) / volatility * np.sqrt(252) if volatility > 0 else 0
        
        # Maximum drawdown
        peak = portfolio_history[0]
        max_drawdown = 0
        
        for value in portfolio_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_drawdown = max(max_drawdown, dd)
        
        # Value at Risk (VaR) 95%
        var_95 = np.percentile(returns, 5)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'var_95': var_95
        } 