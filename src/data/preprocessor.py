"""
Data Preprocessor Module
=======================

Modul ini berisi implementasi kelas DataPreprocessor
untuk mengolah dan menyiapkan data saham.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from .indicators import TechnicalIndicators

class DataPreprocessor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = None
        
    def download_data(self):
        """Mengunduh data saham"""
        try:
            print(f"Downloading data for {self.ticker}...")
            # Pastikan format tanggal benar
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            
            # Download data dengan format yang benar
            self.data = yf.download(
                self.ticker,
                start=start,
                end=end,
                progress=False
            )
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
                
            print(f"Downloaded {len(self.data)} data points")
            
            # Pastikan data dalam format yang benar
            self.data = self.data.astype({
                'Open': 'float64',
                'High': 'float64',
                'Low': 'float64',
                'Close': 'float64',
                'Volume': 'float64'
            })
            
            return True
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            return False
            
    def calculate_indicators(self):
        """Menghitung indikator teknikal"""
        try:
            if self.data is None:
                raise ValueError("No data available. Please download data first.")
                
            # Buat DataFrame baru untuk menyimpan indikator
            indicators = pd.DataFrame(index=self.data.index)
            
            # Moving Averages
            indicators['SMA_14'] = self.data['Close'].rolling(window=14, min_periods=1).mean()
            indicators['SMA_50'] = self.data['Close'].rolling(window=50, min_periods=1).mean()
            indicators['SMA_200'] = self.data['Close'].rolling(window=200, min_periods=1).mean()
            
            # Exponential Moving Averages
            indicators['EMA_9'] = self.data['Close'].ewm(span=9, adjust=False, min_periods=1).mean()
            indicators['EMA_21'] = self.data['Close'].ewm(span=21, adjust=False, min_periods=1).mean()
            indicators['EMA_55'] = self.data['Close'].ewm(span=55, adjust=False, min_periods=1).mean()
            
            # RSI
            delta = self.data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            rs = avg_gain / avg_loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = self.data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = self.data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            indicators['MACD'] = exp1 - exp2
            indicators['Signal_Line'] = indicators['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            indicators['MACD_Histogram'] = indicators['MACD'] - indicators['Signal_Line']
            
            # ADX
            high = self.data['High'].values
            low = self.data['Low'].values
            close = self.data['Close'].values
            indicators['ADX'] = TechnicalIndicators.calculate_adx(high, low, close)
            
            # Volume indicators
            volume_ma = self.data['Volume'].rolling(window=20, min_periods=1).mean()
            indicators['Volume_Ratio'] = self.data['Volume'] / volume_ma.replace(0, np.nan)
            
            # Price momentum
            indicators['Momentum'] = self.data['Close'].pct_change(periods=10)
            indicators['ROC'] = ((self.data['Close'] / self.data['Close'].shift(10)) - 1) * 100
            
            # Gabungkan dengan data asli
            self.data = pd.concat([self.data, indicators], axis=1)
            
            # Hapus NaN values
            self.data.dropna(inplace=True)
            
            # Pilih fitur
            feature_columns = [
                'Close', 'Volume', 'SMA_14', 'SMA_50', 'SMA_200',
                'EMA_9', 'EMA_21', 'EMA_55', 'RSI', 'MACD',
                'Signal_Line', 'MACD_Histogram', 'ADX',
                'Volume_Ratio', 'Momentum', 'ROC'
            ]
            
            # Pastikan semua kolom ada
            available_columns = [col for col in feature_columns if col in self.data.columns]
            self.features = self.data[available_columns]
            
            return True
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return False

    def calculate_obv(self, close, volume):
        """Menghitung On Balance Volume (OBV) secara manual"""
        obv = pd.Series(0.0, index=close.index)
        
        # OBV hari pertama sama dengan volume hari pertama
        obv.iloc[0] = volume.iloc[0]
        
        # Untuk hari-hari berikutnya
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:  # Harga naik
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:  # Harga turun
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:  # Harga tetap
                obv.iloc[i] = obv.iloc[i-1]
            
        return obv

    def calculate_ichimoku(self, high, low, close):
        """Menghitung indikator Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line): (periode 9)
        tenkan_sen_high = high.rolling(window=9).max()
        tenkan_sen_low = low.rolling(window=9).min()
        tenkan_sen = (tenkan_sen_high + tenkan_sen_low) / 2
        
        # Kijun-sen (Base Line): (periode 26)
        kijun_sen_high = high.rolling(window=26).max()
        kijun_sen_low = low.rolling(window=26).min()
        kijun_sen = (kijun_sen_high + kijun_sen_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (periode 52)
        senkou_span_b_high = high.rolling(window=52).max()
        senkou_span_b_low = low.rolling(window=52).min()
        senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): harga penutupan diplot 26 periode ke belakang
        chikou_span = close.shift(-26)
        
        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })

    def calculate_fibonacci_levels(self, data):
        """Menghitung level-level Fibonacci Retracement untuk periode tertentu"""
        # Ambil high dan low dalam range yang dipilih
        price_max = data['High'].max()
        price_min = data['Low'].min()
        
        # Level-level Fibonacci
        diff = price_max - price_min
        level_0 = price_min  # 0%
        level_23_6 = price_min + 0.236 * diff  # 23.6%
        level_38_2 = price_min + 0.382 * diff  # 38.2% 
        level_50_0 = price_min + 0.5 * diff    # 50%
        level_61_8 = price_min + 0.618 * diff  # 61.8%
        level_78_6 = price_min + 0.786 * diff  # 78.6%
        level_100 = price_max  # 100%
        
        # Buat kolom baru untuk setiap level
        fib_levels = pd.DataFrame(index=data.index)
        fib_levels['fib_0'] = level_0
        fib_levels['fib_23_6'] = level_23_6
        fib_levels['fib_38_2'] = level_38_2
        fib_levels['fib_50_0'] = level_50_0
        fib_levels['fib_61_8'] = level_61_8
        fib_levels['fib_78_6'] = level_78_6
        fib_levels['fib_100'] = level_100
        
        # Hitung jarak relatif harga penutupan terhadap level-level Fibonacci
        fib_levels['close_to_fib_ratio'] = (data['Close'] - level_0) / (level_100 - level_0)
        
        return fib_levels 