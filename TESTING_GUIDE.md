# Panduan Pengujian AI Hedge Fund

Dokumen ini menjelaskan cara melakukan pengujian pada sistem AI Hedge Fund dan menafsirkan hasilnya. Pengujian komprehensif dirancang untuk memverifikasi bahwa semua komponen berfungsi dengan baik dan dapat diintegrasikan bersama.

## Prasyarat

Sebelum menjalankan pengujian, pastikan semua dependensi sudah terpasang:

```bash
pip install -r requirements.txt
```

## Menjalankan Pengujian

### 1. Pengujian Komprehensif

Untuk menjalankan semua pengujian sekaligus, gunakan skrip `test_all_features.py`:

```bash
cd scripts
python test_all_features.py
```

Skrip ini akan menguji:
- RiskManager dengan berbagai parameter dan skenario
- MultiAssetPortfolio dengan simulasi transaksi 10 hari
- Backtester dengan berbagai strategi trading
- PPO Agent dengan indikator teknikal
- Sistem terintegrasi yang menggabungkan semua komponen

Hasil pengujian visualisasi akan disimpan di folder `results/`.

### 2. Pengujian Individu

Anda juga dapat menjalankan pengujian untuk komponen tertentu:

#### PPO Agent

```bash
cd scripts
python test_ppo_agent.py
```

#### Backtest dan RiskManager

```bash
cd scripts
python test_hedge_fund.py
```

## Hasil yang Diharapkan

### 1. RiskManager

Dalam pengujian, RiskManager harus:
- Memberikan ukuran posisi yang lebih kecil saat volatilitas meningkat
- Menghasilkan sinyal stop loss saat harga bergerak di luar batasan
- Menghasilkan sinyal trailing stop saat harga berbalik dari level tertinggi/terendah

Contoh output:
```
Position size dengan volatilitas 0.01: 0.2000
Position size dengan volatilitas 0.2: 0.0250
```

### 2. MultiAssetPortfolio

Dalam pengujian, MultiAssetPortfolio harus:
- Mengelola beberapa aset sekaligus
- Mendukung aksi BUY, SELL, SHORT, dan COVER
- Melacak nilai portofolio, PnL, dan metrik kinerja
- Mendukung partial profit/loss taking

Contoh output:
```
Hari 1 - Alokasi: 3 order, Nilai Portofolio: 100000.00
...
Ringkasan Akhir Portfolio:
  Return: 1.16%
  Sharpe Ratio: 14.6786
  Max Drawdown: 0.10%
```

### 3. Backtester

Dalam pengujian, Backtester harus:
- Menjalankan berbagai strategi trading
- Mendukung mode long-only dan long-short
- Menghasilkan metrik kinerja (return, Sharpe ratio, win rate)
- Membuat visualisasi hasil backtest

Beberapa strategi yang diuji:
- Trend Following
- Mean Reversion
- Predictive

### 4. PPO Agent

Dalam pengujian, PPO Agent harus:
- Memproses state dengan indikator teknikal
- Melakukan training pada data historis
- Menghasilkan keputusan trading (buy/sell/hold)
- Mengoptimalkan return menggunakan reinforcement learning

Contoh output:
```
Hasil PPO dengan indikator teknikal:
  Return: 4.44%
  Sharpe Ratio: 0.5322
  Max Drawdown: 19.81%
```

### 5. Sistem Terintegrasi

Dalam pengujian terintegrasi, sistem harus:
- Menggabungkan RiskManager, MultiAssetPortfolio, Backtester, dan PPO Agent
- Menjalankan strategi pada berbagai aset secara simultan
- Menghasilkan dan mengeksekusi sinyal trading
- Menghasilkan metrik kinerja portofolio keseluruhan

Contoh output:
```
Ringkasan Akhir Portofolio:
  Nilai Awal: $100000.00
  Nilai Akhir: $98921.87
  Return: -1.08%
  Sharpe Ratio: -22.0144
  Max Drawdown: 1.08%
```

## Mendiagnosis Error

Jika terjadi error saat pengujian, periksa:

1. **Broadcasting errors**: Biasanya muncul dalam perhitungan array NumPy. Pastikan dimensi array sesuai.

2. **Dimension mismatch**: Pastikan array data yang digunakan untuk plotting memiliki ukuran yang sama.

3. **Training errors**: Pada PPO Agent, error dapat muncul jika state dimensi tidak konsisten atau format data tidak sesuai.

4. **Integration errors**: Pastikan format data antar komponen kompatibel.

## Menginterpretasikan Visualisasi

Hasil pengujian akan menghasilkan beberapa visualisasi di folder `results/`:

1. **multi_asset_portfolio_test.png**: Menunjukkan perubahan nilai portofolio multi-aset dari waktu ke waktu.

2. **backtest_strategies_comparison.png**: Membandingkan kinerja berbagai strategi trading.

3. **ppo_agent_technical.png**: Menampilkan nilai portofolio dan sinyal trading dari PPO Agent.

4. **integrated_system_test.png**: Menampilkan kinerja sistem terintegrasi dan pergerakan harga aset. 