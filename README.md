# Open-Predictor

Aplikasi prediksi harga saham berbasis deep learning dengan fitur PPO trading signals.

## Fitur

- 🚀 Model PatchTST (Patch Time Series Transformer) untuk prediksi harga
- 🤖 PPO (Proximal Policy Optimization) untuk sinyal trading
- 📊 Visualisasi prediksi dan sinyal trading yang interaktif
- 💹 Backtest strategi trading dengan berbagai parameter
- 📈 Optimasi hyperparameter otomatis
- 🔄 Temperature scaling untuk meningkatkan confidence prediksi
- 🧠 Experience replay dan curriculum learning
- ⚖️ Reward shaping dengan penalti untuk trading frequent
- 📉 Adaptive reward scaling
- 🎯 Dynamic entropy untuk eksplorasi

## Instalasi

1. Clone repository:
```bash
git clone https://github.com/cyclocerine/open-predictor.git
cd open-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Penggunaan

### Parameter Prediksi

Parameter Wajib:
- `--ticker`: Kode saham yang akan diprediksi
  - Format: KODE.JK untuk saham Indonesia (contoh: BMRI.JK)
  - Format: KODE untuk saham US (contoh: AAPL)

- `--mode`: Mode operasi aplikasi
  - `predict`: Untuk melakukan prediksi harga
  - `backtest`: Untuk menguji strategi trading

Parameter Opsional:
- `--model`: Jenis model prediksi (default: "patchtst")
  - `patchtst`: Patch Time Series Transformer

- `--start-date`: Tanggal awal data historis
  - Format: YYYY-MM-DD
  - Default: 5 tahun sebelum tanggal sekarang

- `--end-date`: Tanggal akhir data historis
  - Format: YYYY-MM-DD
  - Default: Tanggal sekarang

- `--lookback`: Jumlah hari yang digunakan untuk prediksi
  - Format: angka (hari)
  - Default: 60 hari
  - Semakin besar nilai, semakin banyak data historis yang digunakan
  - Mempengaruhi akurasi dan kecepatan training

- `--forecast-days`: Jumlah hari yang akan diprediksi
  - Format: angka (hari)
  - Default: 30 hari
  - Semakin besar nilai, semakin jauh prediksi ke depan
  - Mempengaruhi akurasi prediksi jangka panjang

- `--tune`: Aktifkan optimasi hyperparameter
  - Flag tanpa nilai
  - Menggunakan Optuna untuk pencarian hyperparameter optimal

- `--tuning-method`: Metode optimasi hyperparameter
  - `grid`: Grid search (lebih lambat, hasil lebih stabil)
  - `hyperband`: Hyperband search (lebih cepat)
  - Default: "grid"

- `--log-dir`: Direktori untuk menyimpan log training
  - Format: path ke direktori
  - Default: Tidak menyimpan log

- `--ppo`: Aktifkan sinyal trading dengan PPO
  - Flag tanpa nilai
  - Menggunakan reinforcement learning untuk generate sinyal

- `--save-results`: Simpan hasil prediksi
  - Flag tanpa nilai
  - Menyimpan plot dan data CSV di folder results/

Parameter Khusus Backtest:
- `--strategy`: Strategi trading yang digunakan
  - `Trend Following`: Mengikuti tren harga
  - `Mean Reversion`: Trading berdasarkan mean reversion
  - `Predictive`: Trading berdasarkan prediksi model
  - `PPO`: Trading menggunakan PPO agent
  - Default: "PPO"

- `--optimize`: Aktifkan optimasi parameter strategi
  - Flag tanpa nilai
  - Mencari parameter optimal untuk strategi trading

- `--initial-balance`: Modal awal untuk backtest
  - Format: angka (Rupiah)
  - Default: 100000000 (100 juta)

### Contoh Penggunaan

1. Prediksi Sederhana:
```bash
python scripts/run_cli.py --ticker BMRI.JK --mode predict
```

2. Prediksi dengan PPO Trading Signals:
```bash
python scripts/run_cli.py --ticker BMRI.JK --model patchtst --mode predict --ppo --save-results
```

3. Backtest dengan Optimasi:
```bash
python scripts/run_cli.py --ticker BMRI.JK --model patchtst --mode backtest --strategy PPO --optimize --initial-balance 100000000
```

4. Prediksi dengan Hyperparameter Tuning:
```bash
python scripts/run_cli.py --ticker BMRI.JK --mode predict --tune --tuning-method hyperband --log-dir logs/
```

5. Prediksi dengan Custom Lookback dan Forecast:
```bash
python scripts/run_cli.py --ticker BMRI.JK --mode predict --lookback 90 --forecast-days 60
```

6. Backtest dengan Custom Lookback:
```bash
python scripts/run_cli.py --ticker BMRI.JK --mode backtest --strategy PPO --lookback 120 --forecast-days 45
```

## Struktur Proyek

```
open-predictor/
├── scripts/
│   └── run_cli.py          # Script utama CLI
├── src/
│   ├── data/               # Modul data processing
│   ├── models/             # Model-model deep learning
│   └── trading/           # Implementasi trading & PPO
├── results/               # Hasil prediksi & backtest
└── requirements.txt      # Dependencies
```

## Model PatchTST

PatchTST (Patch Time Series Transformer) adalah arsitektur transformer yang dioptimalkan untuk time series forecasting:

- Patch embedding untuk menangkap pola temporal
- Self-attention untuk dependencies jangka panjang
- Positional encoding adaptif
- Channel-independent processing

## PPO Trading Agent

PPO (Proximal Policy Optimization) agent untuk menghasilkan sinyal trading:

- Actor-Critic architecture
- Experience replay buffer
- Curriculum learning
- Temperature scaling
- Dynamic entropy adjustment
- Reward shaping

## Metrik Evaluasi

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

## Hasil Trading

- Sinyal: Buy, Sell, Hold
- Confidence level per sinyal
- Ringkasan distribusi sinyal
- Visualisasi prediksi & sinyal
- Backtest performance metrics

## Lisensi

MIT License

## Author

cyclocerine