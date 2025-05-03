"""
Technical Indicators Module
==========================

Modul ini berisi implementasi indikator-indikator teknikal
yang digunakan dalam analisis saham.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

class TechnicalIndicators:
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Menghitung Average Directional Index (ADX)"""
        try:
            # Konversi ke numpy array 1D
            high = np.array(high).flatten()
            low = np.array(low).flatten()
            close = np.array(close).flatten()
            
            # True Range
            high_low = high - low
            high_close_prev = np.abs(high - np.roll(close, 1))
            low_close_prev = np.abs(low - np.roll(close, 1))
            tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            
            # Directional Movement
            up_move = high - np.roll(high, 1)
            down_move = np.roll(low, 1) - low
            
            # +DM dan -DM
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothing
            tr_smooth = np.convolve(tr, np.ones(period)/period, mode='valid')
            pos_dm_smooth = np.convolve(pos_dm, np.ones(period)/period, mode='valid')
            neg_dm_smooth = np.convolve(neg_dm, np.ones(period)/period, mode='valid')
            
            # Directional Indicators
            pos_di = 100 * (pos_dm_smooth / tr_smooth)
            neg_di = 100 * (neg_dm_smooth / tr_smooth)
            
            # ADX
            dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
            adx = np.convolve(dx, np.ones(period)/period, mode='valid')
            
            # Padding
            pad_size = len(high) - len(adx)
            adx = np.pad(adx, (pad_size, 0), mode='edge')
            
            return adx
        except Exception as e:
            print(f"Error in calculate_adx: {str(e)}")
            return np.zeros_like(high)

    @staticmethod
    def calculate_rsi(close, period=14):
        """Menghitung Relative Strength Index (RSI)"""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return np.pad(rsi, (period, 0), mode='edge')

    @staticmethod
    def calculate_macd(close, fast=12, slow=26, signal=9):
        """Menghitung MACD"""
        exp1 = pd.Series(close).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(close).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = pd.Series(macd).ewm(span=signal, adjust=False).mean()
        return macd, signal_line 

def add_technical_indicators(df, include_all=False):
    """
    Tambahkan berbagai indikator teknikal ke DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame dengan setidaknya kolom 'Open', 'High', 'Low', 'Close', 'Volume'
    include_all : bool, optional
        Jika True, tambahkan semua indikator termasuk yang komputasi berat
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan indikator teknikal tambahan
    """
    # Pastikan kolom yang diperlukan ada
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame harus memiliki kolom: {required_columns}")
        
    # Buat copy untuk menghindari warning
    df = df.copy()
    
    # 1. Indikator Trend
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # 2. Indikator Momentum
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    df['Lowest_14'] = df['Low'].rolling(window=14).min()
    df['Highest_14'] = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - df['Lowest_14']) / (df['Highest_14'] - df['Lowest_14']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Rate of Change (ROC)
    df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
    df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
    
    # 3. Indikator Volatilitas
    # Bollinger Bands
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['Middle_Band'] + (df['STD_20'] * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['STD_20'] * 2)
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['Middle_Band']
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # Normalized price with ATR
    df['Price_ATR_Ratio'] = df['Close'] / df['ATR_14']
    
    # 4. Indikator Volume
    # On-Balance Volume (OBV)
    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['Volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    df['OBV'] = obv
    
    # Volume Moving Average
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']
    
    # 5. Statistik Harga
    # Returns
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatilitas Historis
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    # Normalized Price
    df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
    df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']
    
    # 6. Indikator Lainnya (jika include_all=True)
    if include_all:
        # Ichimoku Cloud
        df['Tenkan_Sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Kijun_Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
        df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
        df['Chikou_Span'] = df['Close'].shift(-26)
        
        # Parabolic SAR (simplified implementation)
        df['SAR'] = df['SMA_20'] - (0.02 * df['STD_20'])
        
        # Linear Regression Slope
        df['Slope_20'] = df['Close'].rolling(window=20).apply(
            lambda x: linregress(np.arange(len(x)), x)[0] / x.iloc[0], raw=False)
            
        # Williams %R
        df['Williams_%R'] = -100 * ((df['Highest_14'] - df['Close']) / (df['Highest_14'] - df['Lowest_14']))
        
        # Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        mean_price = typical_price.rolling(window=20).mean()
        mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI'] = (typical_price - mean_price) / (0.015 * mean_deviation)
        
        # Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        pos_flow = pd.Series(np.zeros(len(df)), index=df.index)
        neg_flow = pd.Series(np.zeros(len(df)), index=df.index)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                pos_flow.iloc[i] = money_flow.iloc[i]
            else:
                neg_flow.iloc[i] = money_flow.iloc[i]
        
        pos_mf = pos_flow.rolling(window=14).sum()
        neg_mf = neg_flow.rolling(window=14).sum()
        
        mf_ratio = pos_mf / neg_mf
        df['MFI'] = 100 - (100 / (1 + mf_ratio))
    
    # 7. Market Regime Features
    # Trend Strength (ADX)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff(-1).abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.DataFrame([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ]).max()
    
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    df['ADX'] = dx.rolling(window=14).mean()
    
    # Golden/Death Cross Indicator
    df['SMA_Cross'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    
    # Bollinger Band Squeeze (volatility contraction)
    df['BB_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(window=50).quantile(0.2)
    
    # 8. Normalisasi Data
    # Z-score dari harga (untuk stationary)
    df['Price_Z_Score'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close'].rolling(window=20).std()
    
    # Persentase dari 52-week high/low
    df['52W_High'] = df['Close'].rolling(window=252).max()
    df['52W_Low'] = df['Close'].rolling(window=252).min()
    df['Pct_52W_High'] = df['Close'] / df['52W_High']
    df['Pct_52W_Low'] = df['Close'] / df['52W_Low']
    
    # 9. Bersihkan NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def get_feature_importance():
    """
    Dapatkan daftar fitur penting dan penjelasannya
    
    Returns:
    --------
    dict
        Dictionary dengan nama fitur sebagai key dan penjelasan sebagai value
    """
    return {
        'SMA_50': 'Rata-rata bergerak sederhana 50 hari, indikator trend jangka menengah',
        'SMA_200': 'Rata-rata bergerak sederhana 200 hari, indikator trend jangka panjang',
        'EMA_20': 'Rata-rata bergerak eksponensial 20 hari, lebih responsif terhadap perubahan harga terbaru',
        'MACD': 'Moving Average Convergence Divergence, indikator momentum trend',
        'RSI': 'Relative Strength Index, mengukur kecepatan dan perubahan pergerakan harga (oversold/overbought)',
        'BB_Width': 'Lebar Bollinger Bands, mengukur volatilitas',
        'ATR_14': 'Average True Range, mengukur volatilitas pasar',
        'OBV': 'On-Balance Volume, mengukur aliran volume positif dan negatif',
        'Volatility_20': 'Volatilitas rolling 20 hari (diannualisasi)',
        'ADX': 'Average Directional Index, mengukur kekuatan trend',
        'Price_Z_Score': 'Z-score dari harga, menunjukkan deviasi dari mean periode tertentu',
        'Pct_52W_High': 'Persentase dari 52-week high, indikator momentum/strength'
    }

def calculate_rsi(prices, window=14):
    """
    Hitung Relative Strength Index (RSI)
    
    Parameters:
    -----------
    prices : array-like
        Array harga historis
    window : int, optional
        Periode RSI, default 14
    
    Returns:
    --------
    array
        Array RSI
    """
    # Konversi ke numpy array jika belum
    prices = np.array(prices)
    
    # Hitung perubahan harga
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    
    # Inisialisasi arrays
    up = np.zeros(deltas.shape)
    down = np.zeros(deltas.shape)
    
    # Pisahkan pergerakan naik dan turun
    up[deltas > 0] = deltas[deltas > 0]
    down[-deltas > 0] = -deltas[-deltas > 0]
    
    # Hitung rata-rata bergerak
    up_avg = np.zeros_like(prices)
    down_avg = np.zeros_like(prices)
    rs_values = np.zeros_like(prices)
    rsi_values = np.zeros_like(prices)
    
    # Inisialisasi rata-rata pertama
    up_avg[window] = np.mean(up[:window])
    down_avg[window] = np.mean(down[:window])
    
    # Hitung exponential moving average untuk up dan down
    for i in range(window + 1, len(prices)):
        up_avg[i] = (up_avg[i-1] * (window-1) + up[i-1]) / window
        down_avg[i] = (down_avg[i-1] * (window-1) + down[i-1]) / window
        
        # Hitung RS
        if down_avg[i] == 0:
            rs_values[i] = 100.0
        else:
            rs_values[i] = up_avg[i] / down_avg[i]
            
        # Hitung RSI
        rsi_values[i] = 100.0 - (100.0 / (1.0 + rs_values[i]))
    
    return rsi_values

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Hitung Moving Average Convergence Divergence (MACD)
    
    Parameters:
    -----------
    prices : array-like
        Array harga historis
    fast : int, optional
        Periode EMA cepat, default 12
    slow : int, optional
        Periode EMA lambat, default 26
    signal : int, optional
        Periode signal line, default 9
    
    Returns:
    --------
    tuple
        (macd_line, signal_line, histogram)
    """
    # Konversi ke pandas Series untuk memanfaatkan fungsi EMA
    prices_series = pd.Series(prices)
    
    # Hitung EMAs
    ema_fast = prices_series.ewm(span=fast, adjust=False).mean()
    ema_slow = prices_series.ewm(span=slow, adjust=False).mean()
    
    # MACD Line = FastEMA - SlowEMA
    macd_line = ema_fast - ema_slow
    
    # Signal Line = EMA dari MACD Line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Histogram = MACD Line - Signal Line
    histogram = macd_line - signal_line
    
    return macd_line.values, signal_line.values, histogram.values

def add_price_patterns(df):
    """
    Deteksi pola candlestick dan harga
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame dengan kolom 'Open', 'High', 'Low', 'Close'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan kolom pola harga tambahan
    """
    # Buat copy untuk menghindari warning
    df = df.copy()
    
    # 1. Hammer (Bull reversal) - Kaki bawah panjang, badan kecil di atas
    df['body_size'] = np.abs(df['Close'] - df['Open'])
    df['upper_shadow'] = np.where(df['Close'] > df['Open'],
                                 df['High'] - df['Close'],
                                 df['High'] - df['Open'])
    df['lower_shadow'] = np.where(df['Close'] > df['Open'],
                                 df['Open'] - df['Low'],
                                 df['Close'] - df['Low'])
    
    # Hammer: kaki bawah panjang (setidaknya 2x badan), kaki atas pendek
    df['Hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) &
                   (df['upper_shadow'] < 0.5 * df['body_size']))
    
    # 2. Shooting Star (Bear reversal) - Kaki atas panjang, badan kecil di bawah
    df['Shooting_Star'] = ((df['upper_shadow'] > 2 * df['body_size']) &
                          (df['lower_shadow'] < 0.5 * df['body_size']))
    
    # 3. Doji - Open dan Close sangat dekat
    df['Doji'] = df['body_size'] < 0.1 * (df['High'] - df['Low'])
    
    # 4. Engulfing Patterns
    df['Bullish_Engulfing'] = ((df['Open'].shift(1) > df['Close'].shift(1)) &  # Hari sebelumnya bearish
                              (df['Close'] > df['Open']) &  # Hari ini bullish
                              (df['Open'] <= df['Close'].shift(1)) &  # Open lebih rendah dari close sebelumnya
                              (df['Close'] >= df['Open'].shift(1)))  # Close lebih tinggi dari open sebelumnya
    
    df['Bearish_Engulfing'] = ((df['Open'].shift(1) < df['Close'].shift(1)) &  # Hari sebelumnya bullish
                              (df['Close'] < df['Open']) &  # Hari ini bearish
                              (df['Open'] >= df['Close'].shift(1)) &  # Open lebih tinggi dari close sebelumnya
                              (df['Close'] <= df['Open'].shift(1)))  # Close lebih rendah dari open sebelumnya
    
    # 5. Reversal Patterns
    # Double Top (harga tinggi dua kali berturut-turut dengan pullback)
    df['Relative_High'] = df['High'] > df['High'].shift(1).rolling(window=10).max()
    
    # 6. Support/Resistance Levels
    df['Support_Level'] = df['Low'].rolling(window=20).min()
    df['Resistance_Level'] = df['High'].rolling(window=20).max()
    
    # 7. Price at Support/Resistance
    df['At_Support'] = (df['Low'] - df['Support_Level']).abs() < df['ATR_14'] * 0.5
    df['At_Resistance'] = (df['High'] - df['Resistance_Level']).abs() < df['ATR_14'] * 0.5
    
    return df 