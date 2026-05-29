#!/usr/bin/env python
"""
static_backtest.py — Out-of-Sample (OOS) Backtester untuk BBNI.JK
===================================================================

Script mandiri untuk melakukan OOS testing pada saham BBNI.JK menggunakan
arsitektur P-LSTM (Serial Cascade: LSTM → PatchTST) dengan Continuous Action Space.

Alur Kerja:
1. Muat data CSV multivariate (Close, High, Low, Yield_10Y, USD_IDR, LLM_Sentiment)
2. Normalisasi menggunakan parameter HANYA dari In-Sample (2010-2020)
3. Train P-LSTM pada periode In-Sample
4. Simulasi trading harian pada periode Out-of-Sample (2020-2026)
5. Evaluasi performa: Total Return, CAGR, Sharpe, MDD
6. Plot Equity Curve vs Buy-and-Hold Benchmark

Author: AI Hedge Fund v3.0
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

# Tambahkan root project ke sys.path agar bisa import arsitektur model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.patch_lstm import PatchLSTM, PatchLSTMWrapper


# ═══════════════════════════════════════════════════════════════════════════
# 1. UTILITAS: Tick Size Saham Indonesia (IDX)
# ═══════════════════════════════════════════════════════════════════════════

def round_to_tick_size(price: float) -> float:
    """
    Membulatkan harga ke fraksi harga (tick size) sesuai aturan BEI.
    
    Aturan tick size BEI (per 2 Mei 2016):
    - Harga < Rp 200       : Tick = Rp 1
    - Rp 200 - Rp 500      : Tick = Rp 2
    - Rp 500 - Rp 2,000    : Tick = Rp 5
    - Rp 2,000 - Rp 5,000  : Tick = Rp 10  
    - Harga > Rp 5,000     : Tick = Rp 25
    """
    if price < 200:
        tick = 1
    elif price < 500:
        tick = 2
    elif price < 2000:
        tick = 5
    elif price < 5000:
        tick = 10
    else:
        tick = 25
    return round(price / tick) * tick


# ═══════════════════════════════════════════════════════════════════════════
# 2. LOADING & PERSIAPAN DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(csv_path: str, lookback: int = 60):
    """
    Memuat data CSV dan mempersiapkan fitur multivariate.
    
    Parameters:
    -----------
    csv_path : str
        Path ke file CSV. Harus memiliki kolom:
        Tanggal, Close, High, Low, Yield_10Y, USD_IDR, LLM_Sentiment
    lookback : int
        Panjang window lookback untuk sequence input (default: 60)
        
    Returns:
    --------
    dict dengan kunci:
        'dates_is', 'dates_oos': array tanggal untuk masing-masing periode
        'X_train', 'y_train': data training (In-Sample)
        'X_oos': sequences untuk periode OOS
        'prices_oos': harga Close mentah untuk periode OOS
        'highs_oos', 'lows_oos': harga High/Low mentah untuk ATR
        'scaler': scaler yang di-fit pada In-Sample
    """
    print("[1/6] Memuat data dari CSV...")
    df = pd.read_csv(csv_path, parse_dates=['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)
    
    # Kolom fitur WAJIB (urutan penting — kolom 0 = Close)
    required_cols = ['Close', 'High', 'Low']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan di CSV: {missing}")
    
    # Kolom makro opsional — otomatis deteksi semua yang tersedia
    macro_candidates = [
        'Yield_10Y', 'USD_IDR', 'IHSG', 'EIDO', 'EEM',
        'Crude_Oil_WTI', 'Gold', 'LLM_Sentiment'
    ]
    macro_cols = [c for c in macro_candidates if c in df.columns]
    
    feature_cols = required_cols + macro_cols
    
    # Hitung indikator teknikal tambahan langsung dari data
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['MA_20'] = df['Close'].rolling(20).mean().bfill()
    df['Volatility_21'] = df['Returns'].rolling(21).std().bfill()
    
    # Log returns untuk data makro (lebih stasioner daripada raw price)
    for col in ['IHSG', 'EIDO', 'EEM', 'Crude_Oil_WTI', 'Gold', 'USD_IDR']:
        if col in df.columns:
            df[f'{col}_LogRet'] = np.log(df[col] / df[col].shift(1)).fillna(0)
    
    log_ret_cols = [c for c in df.columns if c.endswith('_LogRet')]
    
    # Update daftar fitur
    feature_cols_extended = feature_cols + ['Returns', 'MA_20', 'Volatility_21'] + log_ret_cols
    
    # Forward fill dan drop NaN
    df[feature_cols_extended] = df[feature_cols_extended].ffill().bfill()
    
    print(f"    Total data points: {len(df)}")
    print(f"    Periode: {df['Tanggal'].iloc[0].date()} s/d {df['Tanggal'].iloc[-1].date()}")
    print(f"    Fitur ({len(feature_cols_extended)}):")
    for i, col in enumerate(feature_cols_extended):
        print(f"      {i:2d}. {col}")
    
    # ── Pembagian Rezim ──
    # In-Sample (IS): 2010-01-01 s/d 2019-12-31
    # Out-of-Sample (OOS): 2020-01-01 s/d akhir dataset
    cutoff_date = pd.Timestamp('2020-01-01')
    
    mask_is = df['Tanggal'] < cutoff_date
    mask_oos = df['Tanggal'] >= cutoff_date
    
    df_is = df[mask_is].copy()
    df_oos = df[mask_oos].copy()
    
    print(f"\n[2/6] Pembagian Rezim:")
    print(f"    In-Sample  (Training): {df_is['Tanggal'].iloc[0].date()} s/d "
          f"{df_is['Tanggal'].iloc[-1].date()} ({len(df_is)} hari)")
    print(f"    Out-of-Sample (Test) : {df_oos['Tanggal'].iloc[0].date()} s/d "
          f"{df_oos['Tanggal'].iloc[-1].date()} ({len(df_oos)} hari)")
    
    # ── Normalisasi (KRITIS: Fit HANYA pada In-Sample) ──
    print("\n[3/6] Normalisasi data (fit HANYA pada In-Sample)...")
    features_is = df_is[feature_cols_extended].values
    features_oos = df_oos[feature_cols_extended].values
    
    scaler = MinMaxScaler()
    scaler.fit(features_is)  # FIT HANYA DI SINI
    
    scaled_is = scaler.transform(features_is)
    scaled_oos = scaler.transform(features_oos)
    
    # ── Buat Sequences untuk Training (In-Sample) ──
    X_train, y_train = [], []
    for i in range(lookback, len(scaled_is)):
        X_train.append(scaled_is[i - lookback:i])
        y_train.append(scaled_is[i, 0])  # Target = Close (kolom 0)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # ── Buat Sequences untuk OOS (perlu lookback data terakhir dari IS) ──
    # Gabungkan tail IS + OOS agar sequence pertama OOS valid
    tail_is = scaled_is[-lookback:]
    combined_oos = np.vstack([tail_is, scaled_oos])
    
    X_oos = []
    for i in range(lookback, len(combined_oos)):
        X_oos.append(combined_oos[i - lookback:i])
    X_oos = np.array(X_oos)
    
    print(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"    X_oos shape  : {X_oos.shape}")
    
    return {
        'dates_is': df_is['Tanggal'].values,
        'dates_oos': df_oos['Tanggal'].values,
        'X_train': X_train,
        'y_train': y_train,
        'X_oos': X_oos,
        'prices_oos': df_oos['Close'].values,
        'highs_oos': df_oos['High'].values,
        'lows_oos': df_oos['Low'].values,
        'scaler': scaler,
        'n_features': len(feature_cols_extended),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAINING MODEL P-LSTM
# ═══════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, checkpoint_path='checkpoint_plstm.pt',
                epochs=50, batch_size=32, force_retrain=False):
    """
    Melatih model P-LSTM pada data In-Sample, atau memuat dari checkpoint.
    
    Returns:
    --------
    PatchLSTMWrapper : model yang sudah terlatih
    """
    n_features = X_train.shape[2]
    
    # Cek apakah checkpoint sudah ada
    if os.path.exists(checkpoint_path) and not force_retrain:
        print(f"\n[4/6] Memuat model dari checkpoint: {checkpoint_path}")
        wrapper = PatchLSTMWrapper(input_dim=n_features)
        wrapper.load(checkpoint_path)
        return wrapper
    
    print(f"\n[4/6] Melatih model P-LSTM (input_dim={n_features}, epochs={epochs})...")
    
    # Split IS menjadi train/val (90/10) untuk early stopping
    split_idx = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    
    wrapper = PatchLSTMWrapper(input_dim=n_features)
    wrapper.fit(
        X_tr, y_tr, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=10,
        verbose=1
    )
    
    # Simpan checkpoint
    wrapper.save(checkpoint_path)
    
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════
# 4. SIMULASI TRADING OOS (DAILY LOOP)
# ═══════════════════════════════════════════════════════════════════════════

def run_oos_backtest(model, X_oos, prices_oos, highs_oos, lows_oos,
                     dates_oos, scaler, initial_capital=100_000_000.0):
    """
    Simulasi trading harian pada periode Out-of-Sample.
    
    Logika:
    1. Model P-LSTM memprediksi harga Close hari berikutnya
    2. Selisih prediksi vs harga saat ini dikonversi menjadi aksi kontinu [-1, 1]
    3. Aksi di-clamp oleh RiskManager (Hard Ceiling 50%)
    4. Eksekusi order dengan Slippage Dinamis dan Tick Size IDX
    
    Returns:
    --------
    dict dengan equity_curve, trades, metrics
    """
    print(f"\n[5/6] Menjalankan simulasi OOS ({len(prices_oos)} hari)...")
    
    n_days = len(prices_oos)
    device = model.device
    
    # ── State Portofolio ──
    cash = initial_capital
    shares = 0
    avg_entry = 0.0
    
    equity_curve = np.zeros(n_days)
    trades = []
    
    # ── ATR Buffer (membutuhkan rolling window) ──
    atr_window = 21
    
    for day in range(n_days):
        current_price = prices_oos[day]
        current_high = highs_oos[day]
        current_low = lows_oos[day]
        portfolio_value = cash + shares * current_price
        
        # ── Hitung ATR 21-hari ──
        if day >= atr_window:
            tr_values = []
            for j in range(day - atr_window, day):
                tr1 = highs_oos[j] - lows_oos[j]
                tr2 = abs(highs_oos[j] - prices_oos[j - 1]) if j > 0 else tr1
                tr3 = abs(lows_oos[j] - prices_oos[j - 1]) if j > 0 else tr1
                tr_values.append(max(tr1, tr2, tr3))
            atr_21 = np.mean(tr_values)
        else:
            atr_21 = current_price * 0.02  # Default 2% jika data belum cukup
        
        # ── Slippage Dinamis ──
        slippage = 0.0005 + (atr_21 / current_price) * 0.1
        
        # ── Prediksi Model (hanya jika data sequence tersedia) ──
        if day < len(X_oos):
            seq = torch.FloatTensor(X_oos[day:day+1]).to(device)
            with torch.no_grad():
                pred_scaled = model.model(seq).cpu().numpy().flatten()[0]
            
            # Inverse transform prediksi (hanya kolom Close)
            n_feat = scaler.n_features_in_
            dummy = np.zeros((1, n_feat))
            dummy[0, 0] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0, 0]
            
            # ── Konversi Prediksi → Aksi Kontinu [-1, 1] ──
            # Logika: selisih persen antara prediksi vs harga saat ini
            pct_diff = (pred_price - current_price) / current_price
            
            # Map ke [-1, 1] dengan sensitivitas ±5%
            raw_action = np.clip(pct_diff / 0.05, -1.0, 1.0)
        else:
            raw_action = 0.0  # HOLD jika tidak ada prediksi
        
        # ── RISK MANAGEMENT: Hard Ceiling 50% ──
        max_ceiling = 0.50
        
        # Volatility scaling: semakin tinggi vol, semakin ketat ceiling
        vol_proxy = atr_21 / current_price
        baseline_vol = 0.02
        vol_scaling = min(1.0, baseline_vol / max(vol_proxy, 1e-8))
        max_ceiling *= vol_scaling
        
        # Drawdown circuit breaker
        peak_value = max(equity_curve[:day+1]) if day > 0 else initial_capital
        peak_value = max(peak_value, initial_capital)
        current_dd = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0.0
        
        if current_dd > 0.075:  # Mulai kurangi di 7.5% DD
            dd_ratio = current_dd / 0.15  # Max DD threshold = 15%
            if dd_ratio >= 1.0:
                max_ceiling *= 0.1
            else:
                scale = 1.0 - (dd_ratio - 0.5) * 1.8
                max_ceiling *= max(0.1, min(1.0, scale))
        
        # Clamp aksi
        action = np.clip(raw_action, -max_ceiling, max_ceiling)
        
        # ── INTERPRETASI AKSI KONTINU ──
        if action > 0.05:
            # === BUY ===
            capital_to_use = cash * min(1.0, action)
            exec_price = round_to_tick_size(current_price * (1 + slippage))
            
            if exec_price > 0 and capital_to_use > exec_price:
                # Lot size IDX = 100 lembar
                shares_to_buy = int(capital_to_use / exec_price / 100) * 100
                
                if shares_to_buy >= 100:
                    cost = shares_to_buy * exec_price
                    fee = cost * 0.0015  # Biaya beli 0.15%
                    total_cost = cost + fee
                    
                    if total_cost <= cash:
                        if shares > 0:
                            avg_entry = (shares * avg_entry + shares_to_buy * exec_price) / (shares + shares_to_buy)
                        else:
                            avg_entry = exec_price
                        
                        shares += shares_to_buy
                        cash -= total_cost
                        
                        trades.append({
                            'date': dates_oos[day],
                            'action': 'BUY',
                            'price': current_price,
                            'exec_price': exec_price,
                            'shares': shares_to_buy,
                            'cost': total_cost,
                            'slippage': slippage,
                            'raw_action': raw_action,
                            'clamped_action': action,
                        })
        
        elif action < -0.05:
            # === SELL ===
            if shares > 0:
                sell_ratio = min(1.0, abs(action))
                shares_to_sell = int(shares * sell_ratio / 100) * 100  # Lot size
                
                if shares_to_sell < 100:
                    shares_to_sell = shares  # Jual semua jika sisa < 1 lot
                
                exec_price = round_to_tick_size(current_price * (1 - slippage))
                revenue = shares_to_sell * exec_price
                fee = revenue * 0.0025  # Biaya jual 0.25% (termasuk pajak)
                net_revenue = revenue - fee
                
                realized_pnl = (exec_price - avg_entry) * shares_to_sell - fee
                
                cash += net_revenue
                shares -= shares_to_sell
                
                if shares <= 0:
                    shares = 0
                    avg_entry = 0.0
                
                trades.append({
                    'date': dates_oos[day],
                    'action': 'SELL',
                    'price': current_price,
                    'exec_price': exec_price,
                    'shares': shares_to_sell,
                    'revenue': net_revenue,
                    'realized_pnl': realized_pnl,
                    'slippage': slippage,
                    'raw_action': raw_action,
                    'clamped_action': action,
                })
        
        # ── Update Equity Curve ──
        equity_curve[day] = cash + shares * current_price
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'final_cash': cash,
        'final_shares': shares,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. PERHITUNGAN METRIKS PERFORMA
# ═══════════════════════════════════════════════════════════════════════════

def calculate_metrics(equity_curve, prices_oos, dates_oos, initial_capital, trades):
    """
    Hitung metrik evaluasi performa backtest.
    
    Returns:
    --------
    dict: Total Return, CAGR, Sharpe Ratio, MDD, dll.
    """
    print("\n[6/6] Menghitung metrik performa...")
    
    final_value = equity_curve[-1]
    total_return_pct = (final_value / initial_capital - 1) * 100
    
    # ── CAGR ──
    n_years = (pd.Timestamp(dates_oos[-1]) - pd.Timestamp(dates_oos[0])).days / 365.25
    if n_years > 0 and final_value > 0:
        cagr = (final_value / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0
    
    # ── Sharpe Ratio (Annualized) ──
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # ── Maximum Drawdown ──
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    mdd = np.max(drawdowns) * 100
    
    # ── Benchmark: Buy and Hold ──
    bnh_return = (prices_oos[-1] / prices_oos[0] - 1) * 100
    bnh_cagr = (prices_oos[-1] / prices_oos[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    
    # ── Trade Statistics ──
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    winning_sells = [t for t in sell_trades if t.get('realized_pnl', 0) > 0]
    win_rate = len(winning_sells) / len(sell_trades) * 100 if sell_trades else 0.0
    
    metrics = {
        'Total Return (%)': f"{total_return_pct:.2f}%",
        'CAGR': f"{cagr*100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.4f}",
        'Max Drawdown (MDD)': f"{mdd:.2f}%",
        'Final Portfolio Value': f"Rp {final_value:,.0f}",
        '': '',  # separator
        'Buy & Hold Return': f"{bnh_return:.2f}%",
        'Buy & Hold CAGR': f"{bnh_cagr*100:.2f}%",
        ' ': '',  # separator
        'Total Trades': f"{len(trades)}",
        'Buy Trades': f"{len(buy_trades)}",
        'Sell Trades': f"{len(sell_trades)}",
        'Win Rate': f"{win_rate:.1f}%",
    }
    
    return metrics, {
        'total_return': total_return_pct,
        'cagr': cagr,
        'sharpe': sharpe,
        'mdd': mdd,
        'bnh_return': bnh_return,
        'n_years': n_years,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. VISUALISASI: EQUITY CURVE VS BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(equity_curve, prices_oos, dates_oos, trades, metrics_raw,
                 initial_capital, save_path='oos_backtest_BBNI.png'):
    """
    Plot grafik Equity Curve vs Buy & Hold Benchmark.
    """
    print(f"\n    Membuat grafik → {save_path}")
    
    dates = pd.to_datetime(dates_oos)
    
    # Normalisasi benchmark ke skala portofolio
    bnh_curve = (prices_oos / prices_oos[0]) * initial_capital
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), 
                              gridspec_kw={'height_ratios': [3, 1, 1]},
                              sharex=True)
    fig.patch.set_facecolor('#0d1117')
    
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ── Panel 1: Equity Curve vs Benchmark ──
    ax1 = axes[0]
    ax1.plot(dates, equity_curve, color='#58a6ff', linewidth=1.5, 
             label=f'P-LSTM Strategy ({metrics_raw["total_return"]:.1f}%)', zorder=3)
    ax1.plot(dates, bnh_curve, color='#f0883e', linewidth=1.2, alpha=0.8,
             label=f'Buy & Hold ({metrics_raw["bnh_return"]:.1f}%)', zorder=2)
    ax1.fill_between(dates, equity_curve, bnh_curve, 
                     where=(equity_curve > bnh_curve),
                     color='#238636', alpha=0.15, label='Outperformance')
    ax1.fill_between(dates, equity_curve, bnh_curve,
                     where=(equity_curve < bnh_curve),
                     color='#da3633', alpha=0.15, label='Underperformance')
    
    # Plot buy/sell markers
    buy_dates = [pd.Timestamp(t['date']) for t in trades if t['action'] == 'BUY']
    sell_dates = [pd.Timestamp(t['date']) for t in trades if t['action'] == 'SELL']
    
    for bd in buy_dates:
        idx = np.searchsorted(dates, bd)
        if idx < len(equity_curve):
            ax1.scatter(bd, equity_curve[idx], color='#3fb950', marker='^', 
                       s=20, zorder=4, alpha=0.7)
    for sd in sell_dates:
        idx = np.searchsorted(dates, sd)
        if idx < len(equity_curve):
            ax1.scatter(sd, equity_curve[idx], color='#f85149', marker='v', 
                       s=20, zorder=4, alpha=0.7)
    
    ax1.set_ylabel('Portfolio Value (Rp)', color='#c9d1d9', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, facecolor='#21262d', 
               edgecolor='#30363d', labelcolor='#c9d1d9')
    ax1.set_title(f'OOS Backtest: BBNI.JK (2020-2026) | P-LSTM + Continuous PPO\n'
                  f'CAGR: {metrics_raw["cagr"]*100:.2f}% | Sharpe: {metrics_raw["sharpe"]:.3f} | '
                  f'MDD: {metrics_raw["mdd"]:.1f}%',
                  color='#f0f6fc', fontsize=13, fontweight='bold', pad=15)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rp {x/1e6:.0f}M'))
    ax1.grid(True, alpha=0.1, color='#30363d')
    
    # ── Panel 2: Drawdown ──
    ax2 = axes[1]
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (running_max - equity_curve) / running_max * 100
    ax2.fill_between(dates, 0, -drawdown, color='#da3633', alpha=0.4)
    ax2.plot(dates, -drawdown, color='#f85149', linewidth=0.8)
    ax2.set_ylabel('Drawdown (%)', color='#c9d1d9', fontsize=10)
    ax2.set_ylim(bottom=-metrics_raw['mdd'] * 1.3, top=2)
    ax2.axhline(y=0, color='#30363d', linewidth=0.5)
    ax2.grid(True, alpha=0.1, color='#30363d')
    
    # ── Panel 3: Harga BBNI.JK ──
    ax3 = axes[2]
    ax3.plot(dates, prices_oos, color='#bc8cff', linewidth=1.0, label='BBNI.JK Close')
    ax3.set_ylabel('Harga (Rp)', color='#c9d1d9', fontsize=10)
    ax3.set_xlabel('Tanggal', color='#c9d1d9', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9, facecolor='#21262d',
               edgecolor='#30363d', labelcolor='#c9d1d9')
    ax3.grid(True, alpha=0.1, color='#30363d')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"    ✓ Grafik disimpan: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN: ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='OOS Backtest BBNI.JK — P-LSTM + Continuous Action Space',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python scripts/static_backtest.py --csv data/BBNI_JK.csv
  python scripts/static_backtest.py --csv data/BBNI_JK.csv --epochs 100 --force-retrain
  python scripts/static_backtest.py --csv data/BBNI_JK.csv --capital 500000000
        """
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='Path ke file CSV data BBNI.JK')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_plstm.pt',
                        help='Path checkpoint model (default: checkpoint_plstm.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Jumlah epoch training (default: 50)')
    parser.add_argument('--capital', type=float, default=100_000_000,
                        help='Modal awal dalam Rupiah (default: 100,000,000)')
    parser.add_argument('--lookback', type=int, default=60,
                        help='Panjang window lookback (default: 60)')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Paksa training ulang meskipun checkpoint tersedia')
    parser.add_argument('--output', type=str, default='oos_backtest_BBNI.png',
                        help='Path file output grafik (default: oos_backtest_BBNI.png)')
    
    args = parser.parse_args()
    
    # ── Header ──
    print("=" * 70)
    print("  OOS BACKTEST: BBNI.JK")
    print("  Arsitektur: P-LSTM (Serial Cascade: LSTM → PatchTST)")
    print("  Action Space: Continuous [-1.0, 1.0] + Hybrid Guardrail")
    print(f"  Modal Awal: Rp {args.capital:,.0f}")
    print("=" * 70)
    
    # Step 1-3: Load dan Persiapan Data
    data = load_and_prepare_data(args.csv, lookback=args.lookback)
    
    # Step 4: Training / Load Model
    model = train_model(
        data['X_train'], data['y_train'],
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        force_retrain=args.force_retrain
    )
    
    # Step 5: Simulasi OOS
    results = run_oos_backtest(
        model=model,
        X_oos=data['X_oos'],
        prices_oos=data['prices_oos'],
        highs_oos=data['highs_oos'],
        lows_oos=data['lows_oos'],
        dates_oos=data['dates_oos'],
        scaler=data['scaler'],
        initial_capital=args.capital,
    )
    
    # Step 6: Hitung Metrik
    metrics, metrics_raw = calculate_metrics(
        equity_curve=results['equity_curve'],
        prices_oos=data['prices_oos'],
        dates_oos=data['dates_oos'],
        initial_capital=args.capital,
        trades=results['trades'],
    )
    
    # ── Cetak Hasil ──
    print("\n" + "=" * 70)
    print("  HASIL BACKTEST OUT-OF-SAMPLE")
    print("=" * 70)
    for key, val in metrics.items():
        if key.strip() == '':
            print(f"  {'─' * 40}")
        else:
            print(f"  {key:<25} {val}")
    print("=" * 70)
    
    # ── Cetak Sample Trades ──
    if results['trades']:
        print(f"\n  Sample Trades (5 pertama):")
        print(f"  {'Tanggal':<12} {'Aksi':<6} {'Harga':<10} {'Eksekusi':<10} "
              f"{'Lembar':<10} {'Raw Act':<10} {'Clamped':<10}")
        print(f"  {'─'*68}")
        for t in results['trades'][:5]:
            date_str = pd.Timestamp(t['date']).strftime('%Y-%m-%d')
            print(f"  {date_str:<12} {t['action']:<6} "
                  f"Rp{t['price']:<8,.0f} Rp{t['exec_price']:<8,.0f} "
                  f"{t['shares']:<10,} {t['raw_action']:<10.4f} {t['clamped_action']:<10.4f}")
    
    # ── Plot Grafik ──
    plot_results(
        equity_curve=results['equity_curve'],
        prices_oos=data['prices_oos'],
        dates_oos=data['dates_oos'],
        trades=results['trades'],
        metrics_raw=metrics_raw,
        initial_capital=args.capital,
        save_path=args.output,
    )
    
    print(f"\n  ✓ Backtest selesai. Grafik: {args.output}")
    print(f"  ✓ Posisi akhir: {results['final_shares']:,} lembar | "
          f"Cash: Rp {results['final_cash']:,.0f}")


if __name__ == '__main__':
    main()
