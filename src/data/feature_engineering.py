"""
Trading Feature Engineering Module
===================================

Modul untuk ekstraksi dan scoring fitur teknikal untuk ML models (PatchTST, PPO).

Features:
- Stochastic RSI calculation
- MACD with ADX validation
- Bollinger Bands scoring
- Volume analysis
- Signal scoring system (0.0 - 1.0 dengan 0.5 = neutral)

Author: Faiq
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union


class TradingFeatureEngineer:
    """
    Ekstraksi fitur teknikal dengan scoring untuk ML models.
    
    Scoring Scale:
        1.00 = Strong Bullish
        0.75 = Bullish
        0.50 = Neutral (No action)
        0.25 = Bearish
        0.00 = Strong Bearish
    
    Trend Direction:
        +1.0 = Bullish
         0.0 = Neutral/Sideways
        -1.0 = Bearish
    """
    
    def __init__(self, ohlcv_df: Optional[pd.DataFrame] = None, 
                 prices: Optional[np.ndarray] = None,
                 volumes: Optional[np.ndarray] = None):
        """
        Initialize feature engineer.
        
        Args:
            ohlcv_df: DataFrame dengan kolom ['Open', 'High', 'Low', 'Close', 'Volume']
            prices: Array harga Close (alternatif jika tidak ada OHLCV)
            volumes: Array volume (opsional, digunakan bersama prices)
        """
        if ohlcv_df is not None:
            self.df = ohlcv_df.copy()
            self.prices = self.df['Close'].values
            self.high = self.df['High'].values
            self.low = self.df['Low'].values
            self.open = self.df['Open'].values
            self.volume = self.df['Volume'].values
            self.has_ohlcv = True
        elif prices is not None:
            self.prices = np.array(prices).flatten()
            self.volume = np.array(volumes).flatten() if volumes is not None else np.ones_like(self.prices)
            self.high = self.prices  # Fallback
            self.low = self.prices   # Fallback
            self.open = self.prices  # Fallback
            self.has_ohlcv = False
            self.df = pd.DataFrame({'Close': self.prices, 'Volume': self.volume})
        else:
            raise ValueError("Harus menyediakan ohlcv_df atau prices")
        
        self.n = len(self.prices)
        self._indicators = {}
        self._scores = {}
    
    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================
    
    def calculate_rsi(self, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        delta = np.diff(self.prices, prepend=self.prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing (EMA)
        alpha = 1 / period
        avg_gain = np.zeros(self.n)
        avg_loss = np.zeros(self.n)
        
        # Initialize
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period + 1, self.n):
            avg_gain[i] = avg_gain[i-1] * (1 - alpha) + gain[i] * alpha
            avg_loss[i] = avg_loss[i-1] * (1 - alpha) + loss[i] * alpha
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50  # Fill initial values with neutral
        
        self._indicators['rsi'] = rsi
        return rsi
    
    def calculate_stochastic_rsi(self, rsi_period: int = 14, 
                                   stoch_period: int = 14, 
                                   k_period: int = 3, 
                                   d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic RSI.
        
        Returns:
            Tuple of (stoch_rsi_k, stoch_rsi_d)
        """
        # Get RSI first
        if 'rsi' not in self._indicators:
            rsi = self.calculate_rsi(rsi_period)
        else:
            rsi = self._indicators['rsi']
        
        # Stochastic on RSI
        rsi_series = pd.Series(rsi)
        lowest_rsi = rsi_series.rolling(window=stoch_period, min_periods=1).min()
        highest_rsi = rsi_series.rolling(window=stoch_period, min_periods=1).max()
        
        # Stochastic RSI %K
        denom = highest_rsi - lowest_rsi
        stoch_rsi = np.where(denom != 0, (rsi - lowest_rsi) / denom * 100, 50)
        
        # Smooth %K and get %D
        stoch_rsi_k = pd.Series(stoch_rsi).rolling(window=k_period, min_periods=1).mean().values
        stoch_rsi_d = pd.Series(stoch_rsi_k).rolling(window=d_period, min_periods=1).mean().values
        
        self._indicators['stoch_rsi_k'] = stoch_rsi_k
        self._indicators['stoch_rsi_d'] = stoch_rsi_d
        
        return stoch_rsi_k, stoch_rsi_d
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD.
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        prices_series = pd.Series(self.prices)
        
        ema_fast = prices_series.ewm(span=fast, adjust=False).mean()
        ema_slow = prices_series.ewm(span=slow, adjust=False).mean()
        
        macd_line = (ema_fast - ema_slow).values
        signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
        histogram = macd_line - signal_line
        
        self._indicators['macd'] = macd_line
        self._indicators['macd_signal'] = signal_line
        self._indicators['macd_histogram'] = histogram
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, period: int = 20, 
                                   std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Returns:
            Tuple of (middle_band, upper_band, lower_band, percent_b)
        """
        prices_series = pd.Series(self.prices)
        
        middle_band = prices_series.rolling(window=period, min_periods=1).mean().values
        std = prices_series.rolling(window=period, min_periods=1).std().values
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # Percent B: Position within bands (0 = lower, 1 = upper)
        band_width = upper_band - lower_band
        percent_b = np.where(band_width != 0, 
                            (self.prices - lower_band) / band_width, 
                            0.5)
        
        # BB Width (normalized by middle band)
        bb_width = np.where(middle_band != 0, band_width / middle_band, 0)
        
        self._indicators['bb_middle'] = middle_band
        self._indicators['bb_upper'] = upper_band
        self._indicators['bb_lower'] = lower_band
        self._indicators['bb_percent_b'] = percent_b
        self._indicators['bb_width'] = bb_width
        
        return middle_band, upper_band, lower_band, percent_b
    
    def calculate_adx(self, period: int = 14) -> np.ndarray:
        """Calculate Average Directional Index (ADX)."""
        if not self.has_ohlcv:
            # Fallback: use volatility-based approximation
            returns = np.diff(self.prices, prepend=self.prices[0]) / np.roll(self.prices, 1)
            returns[0] = 0
            volatility = pd.Series(np.abs(returns)).rolling(window=period, min_periods=1).mean().values
            adx = volatility * 1000  # Scale to 0-100ish range
            adx = np.clip(adx, 0, 100)
            self._indicators['adx'] = adx
            return adx
        
        high = pd.Series(self.high)
        low = pd.Series(self.low)
        close = pd.Series(self.prices)
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = tr.rolling(window=period, min_periods=1).mean()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=1).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=1).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # ADX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = np.where(di_sum != 0, 100 * di_diff / di_sum, 0)
        adx = pd.Series(dx).rolling(window=period, min_periods=1).mean().values
        
        # Handle NaN
        adx = np.nan_to_num(adx, nan=20.0)
        
        self._indicators['adx'] = adx
        self._indicators['plus_di'] = plus_di.values
        self._indicators['minus_di'] = minus_di.values
        
        return adx
    
    def calculate_volume_indicators(self, period: int = 20) -> Dict[str, np.ndarray]:
        """Calculate volume-based indicators."""
        volume_series = pd.Series(self.volume)
        
        # Volume SMA
        volume_sma = volume_series.rolling(window=period, min_periods=1).mean().values
        
        # Volume Ratio
        volume_ratio = np.where(volume_sma != 0, self.volume / volume_sma, 1.0)
        
        # OBV (On-Balance Volume)
        obv = np.zeros(self.n)
        price_change = np.diff(self.prices, prepend=self.prices[0])
        for i in range(1, self.n):
            if price_change[i] > 0:
                obv[i] = obv[i-1] + self.volume[i]
            elif price_change[i] < 0:
                obv[i] = obv[i-1] - self.volume[i]
            else:
                obv[i] = obv[i-1]
        
        # OBV Trend (slope of OBV)
        obv_series = pd.Series(obv)
        obv_sma = obv_series.rolling(window=period, min_periods=1).mean().values
        obv_trend = np.sign(obv - obv_sma)
        
        self._indicators['volume_ratio'] = volume_ratio
        self._indicators['obv'] = obv
        self._indicators['obv_trend'] = obv_trend
        
        return {
            'volume_ratio': volume_ratio,
            'obv': obv,
            'obv_trend': obv_trend
        }
    
    def calculate_log_returns(self) -> np.ndarray:
        """Calculate log returns (stationary transformation)."""
        log_returns = np.log(self.prices / np.roll(self.prices, 1))
        log_returns[0] = 0  # Handle first element
        log_returns = np.nan_to_num(log_returns, nan=0, posinf=0, neginf=0)
        
        self._indicators['log_returns'] = log_returns
        return log_returns
    
    def calculate_rolling_zscore(self, period: int = 20) -> np.ndarray:
        """Calculate rolling Z-score of price."""
        prices_series = pd.Series(self.prices)
        rolling_mean = prices_series.rolling(window=period, min_periods=1).mean()
        rolling_std = prices_series.rolling(window=period, min_periods=1).std()
        
        zscore = np.where(rolling_std != 0, 
                         (self.prices - rolling_mean) / rolling_std, 
                         0)
        zscore = np.nan_to_num(zscore, nan=0)
        
        self._indicators['price_zscore'] = zscore
        return zscore
    
    # =========================================================================
    # SCORING SYSTEM (0.0 - 1.0, 0.5 = Neutral)
    # =========================================================================
    
    def calculate_all_indicators(self):
        """Calculate all indicators at once."""
        self.calculate_rsi()
        self.calculate_stochastic_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_adx()
        self.calculate_volume_indicators()
        self.calculate_log_returns()
        self.calculate_rolling_zscore()
    
    def score_macd(self) -> np.ndarray:
        """
        Score MACD signal with ADX validation.
        
        Returns:
            Array of scores (0.0 - 1.0, 0.5 = neutral)
        """
        if 'macd' not in self._indicators:
            self.calculate_macd()
        if 'adx' not in self._indicators:
            self.calculate_adx()
        
        macd = self._indicators['macd']
        signal = self._indicators['macd_signal']
        histogram = self._indicators['macd_histogram']
        adx = self._indicators['adx']
        
        # Histogram trend (is it increasing?)
        hist_delta = np.diff(histogram, prepend=histogram[0])
        
        scores = np.full(self.n, 0.5)  # Start neutral
        
        for i in range(self.n):
            if macd[i] > signal[i]:  # Bullish
                if hist_delta[i] > 0 and adx[i] > 25:
                    scores[i] = 1.0  # Strong Bullish (trending)
                elif adx[i] > 20:
                    scores[i] = 0.75  # Bullish
                else:
                    scores[i] = 0.6  # Weak Bullish (low trend strength)
            elif macd[i] < signal[i]:  # Bearish
                if hist_delta[i] < 0 and adx[i] > 25:
                    scores[i] = 0.0  # Strong Bearish (trending)
                elif adx[i] > 20:
                    scores[i] = 0.25  # Bearish
                else:
                    scores[i] = 0.4  # Weak Bearish (low trend strength)
            # else: stays at 0.5 (neutral)
        
        self._scores['macd_score'] = scores
        return scores
    
    def score_stochastic_rsi(self) -> np.ndarray:
        """
        Score Stochastic RSI signal with ADX validation.
        
        Returns:
            Array of scores (0.0 - 1.0, 0.5 = neutral)
        """
        if 'stoch_rsi_k' not in self._indicators:
            self.calculate_stochastic_rsi()
        if 'adx' not in self._indicators:
            self.calculate_adx()
        
        k = self._indicators['stoch_rsi_k']
        d = self._indicators['stoch_rsi_d']
        adx = self._indicators['adx']
        
        scores = np.full(self.n, 0.5)  # Start neutral
        
        for i in range(self.n):
            # Overbought/Oversold zones with ADX validation
            if k[i] < 20:  # Oversold
                if k[i] > d[i] and adx[i] > 25:  # Bullish crossover in trend
                    scores[i] = 1.0  # Strong buy signal
                elif k[i] > d[i]:
                    scores[i] = 0.75
                else:
                    scores[i] = 0.6  # Potential reversal
            elif k[i] > 80:  # Overbought
                if k[i] < d[i] and adx[i] > 25:  # Bearish crossover in trend
                    scores[i] = 0.0  # Strong sell signal
                elif k[i] < d[i]:
                    scores[i] = 0.25
                else:
                    scores[i] = 0.4  # Potential reversal
            else:  # Middle zone
                if k[i] > d[i]:
                    scores[i] = 0.6
                elif k[i] < d[i]:
                    scores[i] = 0.4
                # else: neutral 0.5
        
        self._scores['stoch_rsi_score'] = scores
        return scores
    
    def score_bollinger(self) -> np.ndarray:
        """
        Score Bollinger Bands (mean reversion signal).
        
        Returns:
            Array of scores (0.0 - 1.0, 0.5 = neutral)
        """
        if 'bb_percent_b' not in self._indicators:
            self.calculate_bollinger_bands()
        
        percent_b = self._indicators['bb_percent_b']
        
        scores = np.full(self.n, 0.5)  # Start neutral
        
        for i in range(self.n):
            if percent_b[i] < 0:  # Below lower band
                scores[i] = 1.0  # Strong buy (oversold)
            elif percent_b[i] < 0.2:
                scores[i] = 0.75  # Buy
            elif percent_b[i] > 1:  # Above upper band
                scores[i] = 0.0  # Strong sell (overbought)
            elif percent_b[i] > 0.8:
                scores[i] = 0.25  # Sell
            elif 0.4 <= percent_b[i] <= 0.6:
                scores[i] = 0.5  # Neutral (middle of bands)
            elif percent_b[i] < 0.4:
                scores[i] = 0.6  # Slight bullish
            else:
                scores[i] = 0.4  # Slight bearish
        
        self._scores['bb_score'] = scores
        return scores
    
    def score_volume(self) -> np.ndarray:
        """
        Score volume confirmation.
        
        Returns:
            Array of scores (0.0 - 1.0, 0.5 = neutral)
        """
        if 'volume_ratio' not in self._indicators:
            self.calculate_volume_indicators()
        
        volume_ratio = self._indicators['volume_ratio']
        obv_trend = self._indicators['obv_trend']
        
        # Price direction
        price_change = np.diff(self.prices, prepend=self.prices[0])
        price_direction = np.sign(price_change)
        
        scores = np.full(self.n, 0.5)  # Start neutral
        
        for i in range(self.n):
            # High volume confirms price movement
            if volume_ratio[i] > 1.5:  # High volume
                if price_direction[i] > 0 and obv_trend[i] > 0:
                    scores[i] = 1.0  # Strong bullish confirmation
                elif price_direction[i] < 0 and obv_trend[i] < 0:
                    scores[i] = 0.0  # Strong bearish confirmation
                else:
                    scores[i] = 0.5  # Divergence, neutral
            elif volume_ratio[i] > 1.0:  # Above average volume
                if price_direction[i] > 0:
                    scores[i] = 0.65
                elif price_direction[i] < 0:
                    scores[i] = 0.35
            # Low volume: stay neutral
        
        self._scores['volume_score'] = scores
        return scores
    
    def calculate_trend_direction(self) -> np.ndarray:
        """
        Calculate trend direction (-1.0, 0.0, +1.0).
        
        Uses SMA crossover and ADX for confirmation.
        """
        prices_series = pd.Series(self.prices)
        sma_short = prices_series.rolling(window=20, min_periods=1).mean().values
        sma_long = prices_series.rolling(window=50, min_periods=1).mean().values
        
        if 'adx' not in self._indicators:
            self.calculate_adx()
        adx = self._indicators['adx']
        
        trend = np.zeros(self.n)
        
        for i in range(self.n):
            if adx[i] < 20:  # Low trend strength = sideways
                trend[i] = 0.0
            elif sma_short[i] > sma_long[i]:
                trend[i] = 1.0  # Bullish
            elif sma_short[i] < sma_long[i]:
                trend[i] = -1.0  # Bearish
            else:
                trend[i] = 0.0  # Neutral
        
        self._indicators['trend_direction'] = trend
        return trend
    
    def calculate_weekly_trend(self) -> np.ndarray:
        """
        Calculate Weekly Trend and Price deviation.
        Returns array of 'price_to_weekly_ema_ratio'.
        """
        if self.has_ohlcv:
            # Resample to Weekly
            prices_weekly = self.df['Close'].resample('W').last()
            ema10_weekly = prices_weekly.ewm(span=10).mean()
            
            # Reindex to daily (ffill)
            weekly_trend = ema10_weekly.reindex(self.df.index, method='ffill')
            self._indicators['weekly_trend_ratio'] = (self.df['Close'] / weekly_trend - 1).fillna(0).values
        else:
            # Approximation if only prices available (assume 5 days/week)
            ema10_weekly = pd.Series(self.prices).ewm(span=10*5).mean()
            self._indicators['weekly_trend_ratio'] = (self.prices / ema10_weekly - 1).fillna(0).values
            
        return self._indicators['weekly_trend_ratio']

    def get_all_scores(self) -> Dict[str, np.ndarray]:
        """Calculate and return all signal scores."""
        self.score_macd()
        self.score_stochastic_rsi()
        self.score_bollinger()
        self.score_volume()
        self.calculate_trend_direction()
        self.calculate_weekly_trend()  # NEW
        
        return {
            'macd_score': self._scores['macd_score'],
            'stoch_rsi_score': self._scores['stoch_rsi_score'],
            'bb_score': self._scores['bb_score'],
            'volume_score': self._scores['volume_score'],
            'adx_normalized': self._indicators['adx'] / 100,
            'trend_direction': self._indicators['trend_direction'],
            'weekly_trend': self._indicators['weekly_trend_ratio']
        }
    
    # =========================================================================
    # OUTPUT FOR ML MODELS
    # =========================================================================
    
    def get_features_for_ppo(self, step: int) -> np.ndarray:
        """
        Get feature vector for PPO agent at specific time step.
        
        Returns:
            np.ndarray of shape (6,) containing normalized scores
        """
        if not self._scores:
            self.get_all_scores()
        
        features = np.array([
            self._scores['macd_score'][step],
            self._scores['stoch_rsi_score'][step],
            self._scores['bb_score'][step],
            self._scores['volume_score'][step],
            self._indicators['adx'][step] / 100,
            (self._indicators['trend_direction'][step] + 1) / 2,
            np.clip(self._indicators['weekly_trend_ratio'][step] * 10, -1.0, 1.0) # Scaled & Clipped
        ], dtype=np.float32)
        
        return features
    
    def get_features_for_ppo_all(self) -> np.ndarray:
        """
        Get all feature vectors for PPO (full sequence).
        
        Returns:
            np.ndarray of shape (n_steps, 6)
        """
        if not self._scores:
            self.get_all_scores()
        
        features = np.column_stack([
            self._scores['macd_score'],
            self._scores['stoch_rsi_score'],
            self._scores['bb_score'],
            self._scores['volume_score'],
            self._indicators['adx'] / 100,
            (self._indicators['trend_direction'] + 1) / 2,
            np.clip(self._indicators['weekly_trend_ratio'] * 10, -1.0, 1.0)
        ]).astype(np.float32)
        
        return features
    
    def get_features_for_patchtst(self, use_scores: bool = True) -> np.ndarray:
        """
        Get multivariate features for PatchTST.
        
        Args:
            use_scores: If True, include signal scores (more compact)
            
        Returns:
            np.ndarray of shape (seq_len, n_features)
            
        Features:
            - Log Returns (stationary)
            - Rolling Z-Score (20-period)
            - 4 Signal Scores (0-1)
            - ADX normalized
            - BB %B
            - [Optional] 4 Macro Features (if available in df)
        """
        self.calculate_all_indicators()
        self.get_all_scores()
        
        feature_list = []
        
        if use_scores:
            feature_list = [
                self._indicators['log_returns'],
                self._indicators['price_zscore'],
                self._scores['macd_score'],
                self._scores['stoch_rsi_score'],
                self._scores['bb_score'],
                self._scores['volume_score'],
                self._indicators['adx'] / 100,
                np.clip(self._indicators['bb_percent_b'], 0, 1)  # Clip to 0-1
            ]
        else:
            # Raw indicators (more features, potentially more info)
            feature_list = [
                self._indicators['log_returns'],
                self._indicators['price_zscore'],
                self._indicators['rsi'] / 100,
                self._indicators['stoch_rsi_k'] / 100,
                self._indicators['stoch_rsi_d'] / 100,
                np.tanh(self._indicators['macd_histogram'] / np.std(self._indicators['macd_histogram'] + 1e-8)),
                self._indicators['adx'] / 100,
                np.clip(self._indicators['bb_percent_b'], 0, 1),
                np.clip(self._indicators['volume_ratio'] / 3, 0, 1)  # Normalize volume ratio
            ]
            
        # Add Macro Features if available
        macro_cols = ['macro_ixic', 'macro_dji', 'macro_tnx', 'macro_vix']
        if hasattr(self, 'df') and all(col in self.df.columns for col in macro_cols):
             for col in macro_cols:
                 feature_list.append(self.df[col].values)
        
        features = np.column_stack(feature_list).astype(np.float32)
        
        # Handle any remaining NaN
        features = np.nan_to_num(features, nan=0.5)
        
        return features


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_patchtst_features(prices: np.ndarray, 
                              ohlcv_df: Optional[pd.DataFrame] = None,
                              use_scores: bool = True) -> np.ndarray:
    """
    Prepare multivariate STATIONARY features for PatchTST input.
    
    Args:
        prices: Array of Close prices
        ohlcv_df: Optional DataFrame with OHLCV data
        use_scores: If True, use compact scores (8 features)
                   If False, use raw indicators (9 features)
    
    Returns:
        np.ndarray: Shape (seq_len, n_features) ready for PatchTST
        
    Features (Stationarity-aware, use_scores=True):
        - Log Returns
        - Rolling Z-Score (20-period)
        - 4 Signal Scores (normalized 0.0-1.0)
        - ADX normalized
        - BB %B
    """
    engineer = TradingFeatureEngineer(ohlcv_df=ohlcv_df, prices=prices)
    return engineer.get_features_for_patchtst(use_scores=use_scores)


def prepare_ppo_features(prices: np.ndarray,
                         ohlcv_df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Prepare technical features for PPO agent.
    
    Args:
        prices: Array of Close prices
        ohlcv_df: Optional DataFrame with OHLCV data
    
    Returns:
        np.ndarray: Shape (n_steps, 6) technical feature scores
        
    Features (all normalized 0.0-1.0):
        - MACD Score
        - Stochastic RSI Score
        - Bollinger Bands Score
        - Volume Score
        - ADX Normalized
        - Trend Direction (mapped to 0-1)
    """
    engineer = TradingFeatureEngineer(ohlcv_df=ohlcv_df, prices=prices)
    return engineer.get_features_for_ppo_all()


def get_feature_engineer(prices: np.ndarray = None,
                         ohlcv_df: pd.DataFrame = None) -> TradingFeatureEngineer:
    """
    Factory function to create a TradingFeatureEngineer instance.
    
    Args:
        prices: Array of Close prices
        ohlcv_df: DataFrame with OHLCV data
        
    Returns:
        TradingFeatureEngineer instance
    """
    return TradingFeatureEngineer(ohlcv_df=ohlcv_df, prices=prices)


# =============================================================================
# GLOBAL MACRO DATA INTEGRATION
# =============================================================================

def fetch_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch global macro indicators: NASDAQ, Dow Jones, 10-Year Treasury Yield, VIX.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with log returns of macro indicators, shifted by 1 day
        for timezone alignment (US markets close before Asian markets open).
        
    Columns:
        - macro_ixic: NASDAQ log return (shifted)
        - macro_dji: Dow Jones log return (shifted)
        - macro_tnx: 10-Year Treasury log return (shifted)
        - macro_vix: CBOE Volatility Index log return (shifted)
    """
    import yfinance as yf
    
    # Macro tickers (4 features)
    macro_tickers = {
        '^IXIC': 'macro_ixic',   # NASDAQ Composite
        '^DJI': 'macro_dji',     # Dow Jones Industrial Average
        '^TNX': 'macro_tnx',     # 10-Year Treasury Yield
        '^VIX': 'macro_vix'      # CBOE Volatility Index (Fear Gauge)
    }
    
    macro_data = pd.DataFrame()
    
    for ticker, col_name in macro_tickers.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Calculate log returns (stationary transformation)
            log_returns = np.log(data['Close'] / data['Close'].shift(1))
            
            # Shift by 1 day for timezone alignment
            # US market closes before Asian markets open next day
            macro_data[col_name] = log_returns.shift(1)
            
        except Exception as e:
            print(f"[WARN] Failed to fetch {ticker}: {e}")
            # Fill with zeros if fetching fails
            macro_data[col_name] = 0.0
    
    # Fill NaN with forward fill (for different trading calendars)
    macro_data = macro_data.ffill().fillna(0.0)
    
    return macro_data


def prepare_macro_features(ohlcv_df: pd.DataFrame, 
                           start_date: str = None, 
                           end_date: str = None) -> pd.DataFrame:
    """
    Prepare and merge macro features with OHLCV data.
    
    Args:
        ohlcv_df: DataFrame with OHLCV data (must have DatetimeIndex)
        start_date: Override start date (optional)
        end_date: Override end date (optional)
        
    Returns:
        DataFrame with original OHLCV + 4 macro features
    """
    # Infer dates from DataFrame index if not provided
    if start_date is None:
        start_date = ohlcv_df.index.min().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = ohlcv_df.index.max().strftime('%Y-%m-%d')
    
    # Fetch macro data
    macro_df = fetch_macro_data(start_date, end_date)
    
    # Merge with outer join (different trading calendars)
    merged = ohlcv_df.join(macro_df, how='left')
    
    # Forward fill missing macro data (holidays)
    for col in ['macro_ixic', 'macro_dji', 'macro_tnx', 'macro_vix']:
        if col in merged.columns:
            merged[col] = merged[col].ffill().fillna(0.0)
        else:
            merged[col] = 0.0
    
    return merged


def get_macro_regime(macro_features: np.ndarray) -> float:
    """
    Calculate macro regime score based on NASDAQ and DJI returns.
    
    Args:
        macro_features: Array of [macro_ixic, macro_dji, macro_tnx]
        
    Returns:
        Regime score: 
            > 0.5: Risk-On (bullish global sentiment)
            < 0.5: Risk-Off (bearish global sentiment)
            = 0.5: Neutral
    """
    if len(macro_features) < 3:
        return 0.5  # Neutral if data unavailable
    
    ixic_ret = macro_features[0]
    dji_ret = macro_features[1]
    tnx_ret = macro_features[2]
    
    # Average equity return (NASDAQ + DJI)
    equity_sentiment = (ixic_ret + dji_ret) / 2
    
    # Treasury yield rising = Risk-Off (money leaving equities)
    # Treasury yield falling = Risk-On (money entering equities)
    treasury_signal = -tnx_ret  # Inverted
    
    # Combine: 70% equity, 30% treasury
    raw_score = 0.7 * equity_sentiment + 0.3 * treasury_signal
    
    # Normalize to 0-1 range (assuming returns are typically -3% to +3%)
    normalized = np.clip((raw_score + 0.03) / 0.06, 0.0, 1.0)
    
    return float(normalized)

