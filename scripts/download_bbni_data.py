#!/usr/bin/env python
"""
Utility: Download data BBNI.JK + Makro Ekonomi Indonesia
=========================================================

Mengunduh data BBNI.JK beserta data makroekonomi yang relevan untuk
analisis saham perbankan Indonesia:

1. Kurs & Likuiditas Domestik:
   - USD/IDR (USDIDR=X) : Kurs Rupiah
   - IHSG (^JKSE)       : Jakarta Composite Index

2. Proxy Arus Modal Asing:
   - EIDO               : iShares MSCI Indonesia ETF
   - EEM                : MSCI Emerging Markets ETF

3. Komoditas Utama:
   - CL=F               : Crude Oil WTI
   - GC=F               : Gold Futures

4. Yield & Sentimen Global:
   - ^TNX               : US 10-Year Treasury Yield
   - LLM_Sentiment      : Placeholder (0.0) untuk integrasi future
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import yfinance as yf
import numpy as np

# ── Konfigurasi Ticker Makro ──
MACRO_TICKERS = {
    # Ticker yfinance     → Nama kolom di CSV
    'USDIDR=X':            'USD_IDR',
    '^JKSE':               'IHSG',
    'EIDO':                'EIDO',
    'EEM':                 'EEM',
    'CL=F':                'Crude_Oil_WTI',
    'GC=F':                'Gold',
    '^TNX':                'Yield_10Y',
}


def download_single(ticker, start, end, col_name):
    """Download satu ticker dan kembalikan Series Close yang sudah di-rename."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            print(f"    ⚠ {ticker} ({col_name}): Tidak ada data")
            return pd.Series(dtype=float, name=col_name)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        
        series = data['Close'].rename(col_name)
        print(f"    ✓ {ticker} ({col_name}): {len(series)} rows")
        return series
    except Exception as e:
        print(f"    ✗ {ticker} ({col_name}): Error — {e}")
        return pd.Series(dtype=float, name=col_name)


def main():
    START = '2010-01-01'
    END = '2026-06-01'
    
    print("=" * 60)
    print("  DOWNLOAD DATA: BBNI.JK + MAKRO EKONOMI INDONESIA")
    print("=" * 60)
    
    # ── 1. Download BBNI.JK ──
    print("\n[1/3] Mengunduh BBNI.JK...")
    bbni = yf.download('BBNI.JK', start=START, end=END, progress=False)
    
    if bbni.empty:
        print("FATAL: Gagal mengunduh data BBNI.JK")
        return
    
    if isinstance(bbni.columns, pd.MultiIndex):
        bbni.columns = [c[0] if isinstance(c, tuple) else c for c in bbni.columns]
    
    print(f"    ✓ BBNI.JK: {len(bbni)} rows ({bbni.index[0].date()} — {bbni.index[-1].date()})")
    
    # Buat DataFrame utama
    df = pd.DataFrame({
        'Tanggal': bbni.index,
        'Close': bbni['Close'].values,
        'High': bbni['High'].values,
        'Low': bbni['Low'].values,
    }).reset_index(drop=True)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.set_index('Tanggal')
    
    # ── 2. Download Semua Makro Data ──
    print(f"\n[2/3] Mengunduh {len(MACRO_TICKERS)} indikator makroekonomi...")
    
    for ticker, col_name in MACRO_TICKERS.items():
        series = download_single(ticker, START, END, col_name)
        df = df.join(series, how='left')
    
    # ── 3. Post-Processing ──
    print(f"\n[3/3] Post-processing...")
    
    # Forward fill → backward fill (hari libur, missing data)
    for col in MACRO_TICKERS.values():
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Default fallback jika seluruh kolom kosong
    defaults = {
        'USD_IDR': 15000.0,
        'IHSG': 6000.0,
        'EIDO': 25.0,
        'EEM': 40.0,
        'Crude_Oil_WTI': 70.0,
        'Gold': 1800.0,
        'Yield_10Y': 7.0,
    }
    for col, default in defaults.items():
        if col in df.columns and df[col].isna().all():
            df[col] = default
            print(f"    ⚠ {col}: Menggunakan default {default}")
    
    # LLM_Sentiment: placeholder netral
    df['LLM_Sentiment'] = 0.0
    
    # Reset index
    df = df.reset_index()
    
    # Buang baris yang Close-nya NaN (hari libur BBNI tapi data lain ada)
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    
    # ── Simpan ──
    output_path = os.path.join('data', 'BBNI_JK.csv')
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"  ✓ Data disimpan ke: {output_path}")
    print(f"  ✓ Periode: {df['Tanggal'].iloc[0]} — {df['Tanggal'].iloc[-1]}")
    print(f"  ✓ Total rows: {len(df)}")
    print(f"  ✓ Kolom ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        sample = df[col].iloc[-1]
        if col == 'Tanggal':
            print(f"      {i}. {col}")
        else:
            print(f"      {i}. {col:<20} (sample terakhir: {sample:,.2f})")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
