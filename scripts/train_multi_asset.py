#!/usr/bin/env python
"""
Multi-Asset PatchTST Training & Prediction Script (Enhanced)
=============================================================

Features:
- Train on IHSG + S&P 500 + Gold + USD/IDR
- Fine-tune on specific ticker
- OHLC + Technical Indicators as features
- Visualization with actual vs predicted vs forecast

Usage:
    # Train multi-asset model
    python train_multi_asset.py --mode train --epochs 50
    
    # Predict with fine-tuning
    python train_multi_asset.py --mode predict --ticker BBCA.JK --forecast-days 10 --fine-tune
"""

import os
import sys
import time
import argparse
import warnings
import pickle
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.patchtst_model import PatchTST, PatchTSTWrapper

# ============================================================================
# TICKER LISTS
# ============================================================================

SP500_TICKERS = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
    "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
    "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO",
    "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX",
    "BDX", "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BK", "BA",
    "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP",
    "CHRW", "CDNS", "CPT", "CPB", "COF", "CAH", "CCL", "CARR", "CAT", "CBOE",
    "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX",
    "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX",
    "CME", "CMS", "KO", "CTSH", "COIN", "CL", "CMCSA", "CAG", "COP", "ED",
    "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST", "CTRA",
    "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DDOG", "DVA", "DAY",
    "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG", "DLTR",
    "D", "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "ETN",
    "EBAY", "ECL", "EIX", "EW", "EA", "ELV", "EME", "EMR", "ETR", "EOG",
    "EPAM", "EQT", "EFX", "EQIX", "EQR", "ERIE", "ESS", "EL", "EG", "EVRG",
    "ES", "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO",
    "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR", "FE", "FISV", "F", "FTNT",
    "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV",
    "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GDDY",
    "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HPE", "HLT",
    "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN",
    "HII", "IBM", "IEX", "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "IBKR",
    "ICE", "IFF", "IP", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT",
    "JBL", "JKH", "J", "JNJ", "JCI", "JPM", "KVUE", "KDP", "KEY", "KEYS",
    "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX",
    "LW", "LVS", "LDOS", "LEN", "LII", "LLY", "LIN", "LYV", "LKQ", "LMT",
    "L", "LOW", "LULU", "LYB", "MTB", "MPC", "MAR", "MMC", "MLM", "MAS",
    "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD",
    "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ",
    "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX",
    "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC",
    "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXP", "ORLY", "OXY", "ODFL", "OMC",
    "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PSKY", "PH",
    "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW",
    "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG",
    "PTC", "PSA", "PHM", "PWR", "QCOM", "DGX", "Q", "RL", "RJF", "RTX",
    "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "HOOD", "ROK", "ROL",
    "ROP", "ROST", "RCL", "SPGI", "CRM", "SNDK", "SBAC", "SLB", "STX", "SRE",
    "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA", "SOLS", "SOLV", "SO",
    "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS",
    "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER",
    "TSLA", "TXN", "TPL", "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT",
    "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA",
    "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN",
    "VRSK", "VZ", "VRTX", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW",
    "WAB", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST",
    "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM",
    "ZBRA", "ZBH", "ZTS"
]

IHSG_TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK", "BBNI.JK", "ICBP.JK",
    "INDF.JK", "KLBF.JK", "HMSP.JK", "GGRM.JK", "PGAS.JK", "SMGR.JK", "UNTR.JK", "JSMR.JK",
    "CPIN.JK", "PTBA.JK", "ANTM.JK", "ADRO.JK", "ITMG.JK", "INCO.JK", "MEDC.JK", "AKRA.JK",
    "MNCN.JK", "SCMA.JK", "EXCL.JK", "ISAT.JK", "TOWR.JK", "TBIG.JK", "MAPI.JK", "LPPF.JK",
    "SMRA.JK", "BSDE.JK", "PWON.JK", "CTRA.JK", "JPFA.JK", "MAIN.JK", "INTP.JK", "WIKA.JK",
    "PTPP.JK", "WSKT.JK", "ADHI.JK", "TINS.JK", "INDY.JK", "BUMI.JK", "DOID.JK", "HRUM.JK",
    "BRPT.JK", "TPIA.JK", "ESSA.JK", "KEJU.JK", "ACES.JK", "ERAA.JK", "MIKA.JK", "SIDO.JK",
    "KAEF.JK", "HEAL.JK", "SILO.JK", "BIRD.JK", "TAXI.JK", "LSIP.JK", "AALI.JK", "DSNG.JK",
    "SGRO.JK", "SSMS.JK", "SMAR.JK", "BWPT.JK", "MDKA.JK", "MYOR.JK", "ULTJ.JK", "ROTI.JK",
    "CAMP.JK", "GOOD.JK", "INKP.JK", "TKIM.JK", "BYAN.JK", "GEMS.JK", "RAJA.JK", "SRTG.JK",
    "ELSA.JK", "PNLF.JK", "ASRI.JK", "DMAS.JK", "LINK.JK", "MTDL.JK", "EMTK.JK", "KIJA.JK",
    "LPKR.JK", "MLPL.JK", "SRIL.JK", "TOBA.JK", "BMTR.JK", "DNET.JK", "ARTO.JK", "BRIS.JK",
    "AGRO.JK", "BCIC.JK", "BNLI.JK", "MEGA.JK"
]

# International Indices (Japan Nikkei 225, Hong Kong Hang Seng, Germany DAX, US Dow Jones)
NIKKEI_TICKERS = [
    "7203.T", "6758.T", "9984.T", "6861.T", "8306.T", "9432.T", "6902.T", "4502.T",
    "8035.T", "7267.T", "6501.T", "7751.T", "4503.T", "6367.T", "3382.T", "8316.T",
    "9433.T", "4063.T", "6954.T", "7974.T", "8411.T", "9983.T", "6273.T", "6981.T",
    "8766.T", "4568.T", "7733.T", "6702.T", "7201.T", "2914.T"
]

HANG_SENG_TICKERS = [
    "0700.HK", "9988.HK", "3690.HK", "1299.HK", "0941.HK", "2318.HK", "0005.HK", 
    "1398.HK", "0883.HK", "0388.HK", "0016.HK", "0027.HK", "0011.HK", "2388.HK",
    "0066.HK", "0001.HK", "0003.HK", "1038.HK", "0688.HK", "2628.HK"
]

DAX_TICKERS = [
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "MRK.DE", "BAS.DE", "MUV2.DE", "BMW.DE",
    "ADS.DE", "DHL.DE", "AIR.DE", "IFX.DE", "VOW3.DE", "RWE.DE", "HEN3.DE", "DB1.DE",
    "BAYN.DE", "BEI.DE", "EOAN.DE", "FRE.DE"
]

DOW_JONES_TICKERS = [
    "UNH", "GS", "HD", "MSFT", "MCD", "CAT", "V", "AMGN", "BA", "CRM",
    "HON", "TRV", "AXP", "JPM", "IBM", "AAPL", "JNJ", "PG", "CVX", "MRK",
    "WMT", "DIS", "NKE", "KO", "CSCO", "VZ", "DOW", "INTC", "MMM"
]

# Cryptocurrency and Altcoins (using yfinance symbols)
CRYPTO_TICKERS = [
    # Major Cryptocurrencies
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD",
    "DOT-USD", "MATIC-USD", "SHIB-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD",
    "LTC-USD", "UNI-USD", "XLM-USD", "NEAR-USD", "BCH-USD", "APT-USD",
    # DeFi Tokens
    "AAVE-USD", "MKR-USD", "CRV-USD", "COMP-USD", "SNX-USD",
    # Layer 2 & Infrastructure
    "ARB-USD", "OP-USD", "IMX-USD", "FTM-USD", "MANA-USD",
    # Gaming & Metaverse
    "SAND-USD", "AXS-USD", "GALA-USD", "ENJ-USD", "ILV-USD",
    # Other Popular Altcoins
    "FIL-USD", "VET-USD", "ALGO-USD", "EGLD-USD", "HBAR-USD", "ICP-USD", "THETA-USD"
]

# Commodities, Forex, and Indices
OTHER_ASSETS = [
    # Commodities
    "GC=F",      # Gold Futures
    "SI=F",      # Silver Futures
    "CL=F",      # Crude Oil
    "NG=F",      # Natural Gas
    # Forex
    "IDR=X",     # USD/IDR
    "EURUSD=X",  # EUR/USD
    "GBPUSD=X",  # GBP/USD
    "USDJPY=X",  # USD/JPY
    # Major Indices ETFs
    "^GSPC",     # S&P 500
    "^DJI",      # Dow Jones
    "^IXIC",     # NASDAQ
    "^N225",     # Nikkei 225
    "^HSI",      # Hang Seng
    "^GDAXI",    # DAX
]

# ============================================================================
# AVAILABLE INDICES FOR TRAINING (user can select which ones to use)
# ============================================================================
AVAILABLE_INDICES = {
    'sp500': ('S&P 500', SP500_TICKERS),
    'ihsg': ('IHSG Indonesia', IHSG_TICKERS),
    'nikkei': ('Nikkei 225 Japan', NIKKEI_TICKERS),
    'hangseng': ('Hang Seng Hong Kong', HANG_SENG_TICKERS),
    'dax': ('DAX Germany', DAX_TICKERS),
    'dow': ('Dow Jones US', DOW_JONES_TICKERS),
    'crypto': ('Crypto & Altcoins', CRYPTO_TICKERS),
    'other': ('Commodities/Forex/Indices', OTHER_ASSETS),
}

MODEL_PATH = "models/patchtst_multi_asset.pt"
METADATA_PATH = "models/patchtst_multi_asset_metadata.pkl"

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def add_technical_indicators(df):
    """Add technical indicators to OHLCV dataframe."""
    # Make a copy
    data = df.copy()
    
    # Price-based features
    data['returns'] = data['Close'].pct_change()
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    # ATR (Average True Range)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=14).mean()
    
    # Volume features
    if 'Volume' in data.columns:
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    
    # Price position
    data['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-8)
    
    # Volatility
    data['Volatility'] = data['returns'].rolling(window=20).std()
    
    return data

# ============================================================================
# DATA FETCHING WITH CSV CACHE
# ============================================================================

# Cache directory for storing downloaded data
CACHE_DIR = "data/cache"

def get_cache_path(ticker, start_date, end_date):
    """Generate cache file path for a ticker."""
    # Sanitize ticker name for filename
    safe_ticker = ticker.replace("/", "_").replace("=", "_").replace("^", "_").replace(".", "_")
    filename = f"{safe_ticker}_{start_date}_{end_date}.csv"
    return os.path.join(CACHE_DIR, filename)

def load_from_cache(ticker, start_date, end_date):
    """Load data from cache if available."""
    cache_path = get_cache_path(ticker, start_date, end_date)
    
    if os.path.exists(cache_path):
        try:
            data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(data) > 0:
                return data
        except Exception:
            pass
    return None

def save_to_cache(ticker, start_date, end_date, data):
    """Save data to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = get_cache_path(ticker, start_date, end_date)
    
    try:
        data.to_csv(cache_path)
    except Exception as e:
        print(f"  [!] Failed to cache {ticker}: {str(e)[:30]}")

def fetch_ohlcv_data(ticker, start_date, end_date, min_days=60, use_cache=True):
    """Fetch OHLCV data with technical indicators. Uses CSV cache if available."""
    
    # Try to load from cache first
    if use_cache:
        cached_data = load_from_cache(ticker, start_date, end_date)
        if cached_data is not None and len(cached_data) >= min_days:
            dates = cached_data.index.tolist()
            return cached_data, dates
    
    # Fetch from yfinance if not in cache
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) < min_days:
            return None, None
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Store dates
        dates = data.index.tolist()
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Drop NaN rows
        data = data.dropna()
        dates = dates[-len(data):]
        
        # Save to cache
        if use_cache and len(data) >= min_days:
            save_to_cache(ticker, start_date, end_date, data)
        
        return data, dates
    except Exception as e:
        print(f"  [!] Error fetching {ticker}: {str(e)[:50]}")
        return None, None

def fetch_all_data_ohlcv(tickers, start_date, end_date, category_name="Assets", use_cache=True):
    """Fetch OHLCV data for all tickers with cache support."""
    print(f"\n[INFO] Fetching {category_name} data...")
    print(f"  Total tickers: {len(tickers)}")
    if use_cache:
        print(f"  Cache dir: {CACHE_DIR}")
    
    all_data = {}
    successful = 0
    failed = 0
    from_cache = 0
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(tickers)}...")
        
        # Check if cached before fetching
        cached_data = load_from_cache(ticker, start_date, end_date) if use_cache else None
        is_cached = cached_data is not None and len(cached_data) >= 60
        
        df, dates = fetch_ohlcv_data(ticker, start_date, end_date, use_cache=use_cache)
        if df is not None:
            all_data[ticker] = {'data': df, 'dates': dates}
            successful += 1
            if is_cached:
                from_cache += 1
        else:
            failed += 1
        
        # Only sleep if we actually fetched from yfinance (not cache)
        if not is_cached:
            time.sleep(0.1)
    
    print(f"  [OK] Successfully loaded: {successful} (from cache: {from_cache})")
    print(f"  [!] Failed: {failed}")
    
    return all_data

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Feature columns to use
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'returns', 
    'SMA_5', 'SMA_10', 'SMA_20', 'MACD', 'MACD_Signal',
    'RSI', 'BB_Width', 'ATR', 'Price_Position', 'Volatility'
]

def prepare_sequences_multi_feature(data_dict, lookback=60, max_samples_per_ticker=2000):
    """Prepare sequences with multiple features.
    
    Args:
        data_dict: Dictionary of ticker data
        lookback: Number of days to look back
        max_samples_per_ticker: Maximum samples per ticker to prevent memory issues
                               Set to None or 0 for unlimited
    """
    all_X = []
    all_y = []
    
    total_tickers = len(data_dict)
    processed = 0
    
    print(f"[INFO] Preparing sequences from {total_tickers} tickers...")
    if max_samples_per_ticker:
        print(f"  Max samples per ticker: {max_samples_per_ticker}")
    
    for ticker, item in data_dict.items():
        processed += 1
        if processed % 100 == 0:
            print(f"  Processing {processed}/{total_tickers}...")
        
        df = item['data']
        
        # Check if all required columns exist
        available_cols = [c for c in FEATURE_COLS if c in df.columns]
        if len(available_cols) < 5:
            continue
        
        if len(df) < lookback + 1:
            continue
        
        # Get feature matrix
        features = df[available_cols].values
        close_prices = df['Close'].values
        
        # Normalize each feature independently
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Normalize close for target
        close_scaler = MinMaxScaler()
        scaled_close = close_scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()
        
        # Create sequences
        ticker_sequences = []
        for i in range(lookback, len(scaled_features)):
            X = scaled_features[i-lookback:i]
            y = scaled_close[i]
            ticker_sequences.append((X, y))
        
        # Sample if too many sequences (to prevent memory issues)
        if max_samples_per_ticker and len(ticker_sequences) > max_samples_per_ticker:
            # Random sample to keep memory down
            indices = np.random.choice(len(ticker_sequences), max_samples_per_ticker, replace=False)
            ticker_sequences = [ticker_sequences[i] for i in indices]
        
        # Add to master list
        for X, y in ticker_sequences:
            all_X.append(X)
            all_y.append(y)
    
    print(f"  [OK] Created {len(all_X)} total sequences")
    
    X = np.array(all_X)
    y = np.array(all_y)
    
    return X, y

def prepare_single_ticker_data(ticker, lookback=60, years=5):
    """Prepare data for single ticker with features."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
    
    print(f"[INFO] Fetching OHLCV data for {ticker}...")
    df, dates = fetch_ohlcv_data(ticker, start_date, end_date, min_days=lookback+1)
    
    if df is None:
        print(f"[ERROR] Failed to fetch data for {ticker}")
        return None, None, None, None, None
    
    print(f"  [OK] Downloaded {len(df)} data points with {len(FEATURE_COLS)} features")
    
    # Get available features
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    # Get feature matrix
    features = df[available_cols].values
    close_prices = df['Close'].values
    
    # Scalers
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    
    close_scaler = MinMaxScaler()
    scaled_close = close_scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_list = []
    y_list = []
    for i in range(lookback, len(scaled_features)):
        X = scaled_features[i-lookback:i]
        y = scaled_close[i]
        X_list.append(X)
        y_list.append(y)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Adjust dates to match sequences
    seq_dates = dates[lookback:]
    
    return X, y, close_scaler, close_prices, seq_dates

# ============================================================================
# TRAINING
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=1e-3):
    """Train PatchTST model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Training on device: {device}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Input features: {X_train.shape[2]}")
    
    model = PatchTSTWrapper(
        input_dim=X_train.shape[2],  # Number of features
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        lr=lr
    )
    
    print(f"\n[INFO] Training for {epochs} epochs...")
    model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model

def fine_tune_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=5e-4):
    """Fine-tune model on specific ticker with aggressive training."""
    print(f"\n[INFO] Fine-tuning for {epochs} epochs (lr={lr})...")
    
    # Use all data for fine-tuning (not just train set)
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    
    # Split 90/10 for fine-tuning (use more data for training)
    ft_split = int(len(X_all) * 0.9)
    X_ft_train, X_ft_val = X_all[:ft_split], X_all[ft_split:]
    y_ft_train, y_ft_val = y_all[:ft_split], y_all[ft_split:]
    
    model.lr = lr
    model.fit(X_ft_train, y_ft_train, X_ft_val, y_ft_val, epochs=epochs, batch_size=16, verbose=1)
    
    return model

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_prediction_chart(ticker, dates, actual_prices, predicted_prices, 
                           forecast_dates, forecast_prices, metrics, save_path):
    """Create simple prediction chart like the user reference image."""
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert dates to day numbers for x-axis (like the reference image)
    n_actual = len(actual_prices)
    x_actual = np.arange(n_actual)
    
    # Plot Actual prices (blue solid line)
    ax.plot(x_actual, actual_prices, 'b-', label='Aktual', linewidth=1.5)
    
    # Plot Predicted prices (red dashed line)
    if len(predicted_prices) > 0:
        # Align prediction with actual (both should have same length after lookback)
        n_pred = len(predicted_prices)
        x_pred = np.arange(n_actual - n_pred, n_actual)
        ax.plot(x_pred, predicted_prices, 'r--', label='Prediksi', linewidth=1.5)
    
    # Plot Forecast (green dash-dot line with markers)
    if len(forecast_prices) > 0:
        x_forecast = np.arange(n_actual, n_actual + len(forecast_prices))
        ax.plot(x_forecast, forecast_prices, 'g-.', label='Forecast', linewidth=2, 
                marker='o', markersize=4)
    
    # Title and labels in Indonesian
    ax.set_title(f'{ticker} - Prediksi Harga Saham dengan PATCHTST', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hari', fontsize=12)
    ax.set_ylabel('Harga', fontsize=12)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Chart saved to: {save_path}")

# ============================================================================
# PREDICTION
# ============================================================================

def predict_ticker(ticker, forecast_days=10, lookback=60, fine_tune=False, fine_tune_epochs=20):
    """Predict with evaluation and visualization."""
    
    print("=" * 60)
    print(f"  PREDICTING: {ticker}")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        print("[INFO] Run training first: python train_multi_asset.py --mode train")
        return None
    
    # Load model
    print(f"\n[INFO] Loading model from {MODEL_PATH}...")
    model = PatchTSTWrapper(
        input_dim=len(FEATURE_COLS),
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        lr=1e-3
    )
    model.load(MODEL_PATH)
    print("  [OK] Model loaded")
    
    # Prepare data
    X, y, scaler, prices, dates = prepare_single_ticker_data(ticker, lookback=lookback)
    if X is None:
        return None
    
    # Train/val split for fine-tuning and evaluation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Fine-tune if requested
    if fine_tune:
        print(f"\n[INFO] Fine-tuning on {ticker}...")
        model = fine_tune_model(model, X_train, y_train, X_val, y_val, 
                                epochs=fine_tune_epochs, lr=5e-4)
        
        # Save fine-tuned model
        ft_model_path = f"models/patchtst_{ticker.replace('.', '_')}_finetuned.pt"
        model.save(ft_model_path)
        print(f"  [OK] Fine-tuned model saved to: {ft_model_path}")
    
    # Evaluation on FULL dataset (train + val) for realistic metrics after fine-tuning
    print(f"\n[INFO] Evaluating on full dataset...")
    y_pred_full = model.predict(X)
    
    # Calculate metrics on full data
    mse = np.mean((y_pred_full - y) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_full - y))
    
    ss_res = np.sum((y - y_pred_full) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Price metrics
    y_actual_prices = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_pred_prices = scaler.inverse_transform(y_pred_full.reshape(-1, 1)).flatten()
    
    mse_price = np.mean((y_pred_prices - y_actual_prices) ** 2)
    rmse_price = np.sqrt(mse_price)
    mae_price = np.mean(np.abs(y_pred_prices - y_actual_prices))
    
    # Direction accuracy
    actual_direction = np.diff(y_actual_prices) > 0
    pred_direction = np.diff(y_pred_prices) > 0
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Print metrics
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"\n  Evaluation samples: {len(y_val)}")
    print(f"\n  [Normalized Metrics]")
    print(f"    MSE:  {mse:.6f}")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    MAE:  {mae:.6f}")
    print(f"    R2:   {r2:.4f}")
    print(f"\n  [Price Metrics]")
    print(f"    MSE:  {mse_price:,.2f}")
    print(f"    RMSE: {rmse_price:,.2f}")
    print(f"    MAE:  {mae_price:,.2f}")
    print(f"\n  [Direction Accuracy]: {direction_accuracy:.1f}%")
    
    # Future forecast
    print(f"\n[INFO] Making predictions for {forecast_days} days...")
    
    last_sequence = X[-1:]
    forecasts = []
    current_sequence = last_sequence.copy()
    
    for i in range(forecast_days):
        pred = model.predict(current_sequence)
        forecasts.append(pred[0])
        
        # Roll sequence and update
        new_seq = np.roll(current_sequence, -1, axis=1)
        # Update Close price column (index 3 in FEATURE_COLS)
        new_seq[0, -1, 3] = pred[0]
        current_sequence = new_seq
    
    forecasts = np.array(forecasts)
    forecast_prices = scaler.inverse_transform(forecasts.reshape(-1, 1)).flatten()
    
    # Create forecast dates
    last_date = dates[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')
    
    last_price = prices[-1]
    total_change = (forecast_prices[-1] - last_price) / last_price * 100
    trend = "BULLISH" if total_change > 0 else "BEARISH"
    
    # Print forecast
    print("\n" + "=" * 60)
    print("  FORECAST RESULTS")
    print("=" * 60)
    print(f"\n  Ticker: {ticker}")
    print(f"  Last Price: {last_price:,.2f}")
    print(f"\n  Forecast for next {forecast_days} days:")
    print("-" * 50)
    print(f"  {'Day':<6} {'Date':<12} {'Price':>12} {'Change':>10}")
    print("-" * 50)
    
    prev_price = last_price
    for i, (date, price) in enumerate(zip(forecast_dates, forecast_prices), 1):
        change = (price - prev_price) / prev_price * 100
        trend_sym = "[+]" if change > 0 else "[-]"
        print(f"  {i:<6} {date.strftime('%Y-%m-%d'):<12} {price:>12,.2f} {trend_sym} {abs(change):>6.2f}%")
        prev_price = price
    
    print("-" * 50)
    print(f"\n  Overall Trend: {trend} ({total_change:+.2f}%)")
    
    # Create visualization
    os.makedirs("results", exist_ok=True)
    
    # Get all predictions on full dataset for chart
    all_predictions = model.predict(X)
    all_pred_prices = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
    
    # Actual prices (aligned with sequences)
    aligned_actual = prices[lookback:]
    aligned_dates = dates
    
    metrics_dict = {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'mse_price': mse_price, 'rmse_price': rmse_price, 'mae_price': mae_price,
        'direction_accuracy': direction_accuracy,
        'trend': trend, 'total_change': total_change
    }
    
    chart_path = f"results/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_chart.png"
    create_prediction_chart(
        ticker=ticker,
        dates=aligned_dates,
        actual_prices=aligned_actual,
        predicted_prices=all_pred_prices,
        forecast_dates=forecast_dates,
        forecast_prices=forecast_prices,
        metrics=metrics_dict,
        save_path=chart_path
    )
    
    # Save results
    results = {
        'ticker': ticker,
        'last_price': last_price,
        'forecast_days': forecast_days,
        'forecast_prices': forecast_prices.tolist(),
        'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'total_change_pct': total_change,
        'fine_tuned': fine_tune,
        'evaluation_metrics': metrics_dict
    }
    
    results_path = f"results/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  [OK] Results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("  ** PREDICTION COMPLETE **")
    print("=" * 60)
    
    return results

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def run_training(epochs=50, batch_size=64, lr=1e-3, lookback=60, selected_indices=None, max_samples_per_ticker=2000):
    """Run full training on multi-asset OHLCV data.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        lookback: Number of days to look back
        selected_indices: List of index keys to use (e.g., ['sp500', 'ihsg', 'crypto'])
        max_samples_per_ticker: Max sequences per ticker (reduce to save memory)
    """
    
    print("=" * 60)
    print("  MULTI-ASSET PATCHTST TRAINING (OHLCV + Technical)")
    print("=" * 60)
    
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = '2000-01-01'  # Use ALL available data for training
    
    # Determine which indices to use
    if selected_indices is None or len(selected_indices) == 0 or 'all' in selected_indices:
        indices_to_use = list(AVAILABLE_INDICES.keys())
    else:
        indices_to_use = [idx for idx in selected_indices if idx in AVAILABLE_INDICES]
    
    print(f"\n[INFO] Parameters:")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Lookback: {lookback} days")
    print(f"  Epochs: {epochs}")
    print(f"  Max samples/ticker: {max_samples_per_ticker}")
    print(f"  Selected indices: {', '.join(indices_to_use)}")
    
    os.makedirs("models", exist_ok=True)
    
    # Fetch data for selected indices
    print("\n" + "=" * 60)
    print("  STEP 1: FETCHING OHLCV DATA")
    print("=" * 60)
    
    all_data = {}
    for idx_key in indices_to_use:
        if idx_key in AVAILABLE_INDICES:
            idx_name, idx_tickers = AVAILABLE_INDICES[idx_key]
            idx_data = fetch_all_data_ohlcv(idx_tickers, START_DATE, END_DATE, idx_name)
            all_data.update(idx_data)
    
    print(f"\n[INFO] Total assets with valid data: {len(all_data)}")
    
    if len(all_data) == 0:
        print("[ERROR] No valid data fetched.")
        return
    
    # Prepare sequences
    print("\n" + "=" * 60)
    print("  STEP 2: PREPARING TRAINING DATA")
    print("=" * 60)
    
    X, y = prepare_sequences_multi_feature(all_data, lookback=lookback, 
                                           max_samples_per_ticker=max_samples_per_ticker)
    print(f"[INFO] Total sequences: {len(X)}")
    print(f"[INFO] Feature dimensions: {X.shape}")
    
    # Train/val split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train
    print("\n" + "=" * 60)
    print("  STEP 3: TRAINING MODEL")
    print("=" * 60)
    
    model = train_model(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, lr=lr)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("  STEP 4: EVALUATION")
    print("=" * 60)
    
    y_pred = model.predict(X_val)
    mse = np.mean((y_pred - y_val) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_val))
    
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n[RESULTS]")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.4f}")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"\n[OK] Model saved to: {MODEL_PATH}")
    
    # Save metadata
    metadata = {
        'tickers': list(all_data.keys()),
        'features': FEATURE_COLS,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'lookback': lookback,
        'total_samples': len(X),
        'metrics': {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    }
    
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"[OK] Metadata saved to: {METADATA_PATH}")
    
    print("\n" + "=" * 60)
    print("  ** TRAINING COMPLETE **")
    print("=" * 60)

# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    # Build list of available indices for help text
    indices_list = ', '.join(AVAILABLE_INDICES.keys())
    
    parser = argparse.ArgumentParser(
        description="Multi-Asset PatchTST with OHLCV + Technical Indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Indices:
  sp500     - S&P 500 (100 stocks)
  ihsg      - IHSG Indonesia (100 stocks)
  nikkei    - Nikkei 225 Japan (30 stocks)
  hangseng  - Hang Seng Hong Kong (20 stocks)
  dax       - DAX Germany (20 stocks)
  dow       - Dow Jones US (29 stocks)
  crypto    - Crypto & Altcoins (42 tokens)
  other     - Commodities/Forex/Indices (14 assets)
  all       - All indices (default)

Examples:
  # Train only on IHSG and S&P 500
  python train_multi_asset.py --mode train --indices sp500 ihsg --epochs 50
  
  # Train on crypto only
  python train_multi_asset.py --mode train --indices crypto --epochs 30
  
  # Train on all indices
  python train_multi_asset.py --mode train --indices all --epochs 50
  
  # Predict with fine-tuning
  python train_multi_asset.py --mode predict --ticker BBCA.JK --fine-tune --forecast-days 20
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='predict')
    parser.add_argument('--indices', type=str, nargs='+', default=['all'],
                        help=f'Indices to use for training: {indices_list}, or "all" (default: all)')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Max samples per ticker to prevent memory issues (default: 2000, 0=unlimited)')
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--forecast-days', type=int, default=10)
    parser.add_argument('--fine-tune', action='store_true', help='Fine-tune on specific ticker')
    parser.add_argument('--fine-tune-epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lookback', type=int, default=60)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Convert max_samples 0 to None for unlimited
    max_samples = args.max_samples if args.max_samples > 0 else None
    
    if args.mode == 'train':
        run_training(
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            lr=args.lr, 
            lookback=args.lookback,
            selected_indices=args.indices,
            max_samples_per_ticker=max_samples
        )
    else:
        predict_ticker(
            ticker=args.ticker,
            forecast_days=args.forecast_days,
            lookback=args.lookback,
            fine_tune=args.fine_tune,
            fine_tune_epochs=args.fine_tune_epochs
        )

if __name__ == "__main__":
    main()
