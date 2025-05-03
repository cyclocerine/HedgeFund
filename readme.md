# 🤖💹 AI Hedge Fund

Sistem kecerdasan buatan untuk manajemen portofolio dan perdagangan otomatis berbasis machine learning.

## 📋 Deskripsi

AI Hedge Fund adalah platform trading algoritmik canggih yang menggabungkan manajemen risiko adaptif, perdagangan multi-aset, model machine learning, dan strategi perdagangan otomatis. Sistem ini dirancang untuk mengoptimalkan keputusan investasi melalui kombinasi dari algoritma trading tradisional dan teknik kecerdasan buatan yang canggih.

## 🌟 Fitur Utama

- **Manajemen Risiko Adaptif**: Mengelola risiko portofolio dengan stop loss dinamis, trailing stop, pengelolaan posisi, dan analisis drawdown.
- **Trading Multi-Aset**: Mendukung perdagangan beberapa aset secara simultan dengan alokasi kapital berbasis risiko.
- **Dukungan Short Selling**: Kemampuan mengambil posisi short dan long untuk memanfaatkan pasar naik maupun turun.
- **Integrasi Model Machine Learning**: PPO (Proximal Policy Optimization) agent untuk pengambilan keputusan trading otomatis.
- **Indikator Teknikal Tingkat Lanjut**: Lebih dari 30 indikator teknikal untuk analisis pasar komprehensif.
- **Backtest Komprehensif**: Kemampuan simulasi historis dengan berbagai strategi dan metrik kinerja.
- **Visualisasi Hasil**: Grafik dan plot interaktif untuk evaluasi performa.

## 🧪 Pengujian

Sistem ini telah menjalani pengujian komprehensif untuk memverifikasi fungsionalitas dan kinerja seluruh komponen:

- **Pengujian RiskManager**: Berhasil menguji berbagai skenario stop loss, trailing stop, dan perhitungan ukuran posisi berdasarkan volatilitas.
- **Pengujian MultiAssetPortfolio**: Simulasi 10 hari trading dengan berbagai aksi buy/sell/short/cover dan alokasi parsial, menghasilkan return positif 1.16%.
- **Pengujian Backtester**: Evaluasi berbagai strategi (Trend Following, Mean Reversion, Predictive) dalam mode long-only dan long-short.
- **Pengujian PPO Agent**: Melatih dan melakukan backtest PPO agent dengan indikator teknikal, menghasilkan return 4.44% dan Sharpe Ratio 0.53.
- **Pengujian Terintegrasi**: Simulasi sistem penuh dengan multiple strategi dan PPO agent untuk trading multi-aset.

Untuk melakukan pengujian sendiri, lihat [TESTING_GUIDE.md](TESTING_GUIDE.md) untuk instruksi lengkap. Laporan bug dan perbaikan tersedia di [BUGFIX_REPORT.md](BUGFIX_REPORT.md).

### 📊 Hasil Pengujian

Beberapa visualisasi hasil pengujian:

![Portfolio Performance](results/multi_asset_portfolio_test.png)
*Kinerja portofolio multi-aset selama periode pengujian*

![Strategies Comparison](results/backtest_strategies_comparison.png)
*Perbandingan berbagai strategi trading*

![PPO Agent Performance](results/ppo_agent_technical.png)
*Kinerja PPO Agent dengan indikator teknikal*

## 🛠️ Instalasi

1. Clone repositori:
   ```bash
   git clone https://github.com/cyclocerine/hedgefund.git
   cd hedgefund
   ```

2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Cara Penggunaan

### GUI

Untuk menjalankan aplikasi GUI:

```bash
cd scripts
python run_app.py
```

### CLI

Untuk menjalankan via command line:

```bash
cd scripts
python run_cli.py --ticker AAPL --days 30
```

### Menjalankan Backtest

```bash
cd scripts
python test_hedge_fund.py
```

## 🧩 Struktur Proyek

```
.
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Utilitas loading data
│   │   ├── indicators.py       # Perhitungan indikator teknikal
│   │   └── preprocessor.py     # Preprocessing data
│   ├── models/
│   │   ├── lstm.py             # Model LSTM untuk prediksi
│   │   ├── regression.py       # Model regresi linear
│   │   └── ensemble.py         # Model ensemble
│   ├── trading/
│   │   ├── backtest.py         # Engine backtesting
│   │   ├── strategies.py       # Strategi trading
│   │   ├── risk_manager.py     # Manajemen risiko portofolio
│   │   ├── portfolio.py        # Pengelolaan portofolio multi-aset
│   │   └── ppo_agent.py        # Agent PPO untuk trading
│   └── utils/
│       ├── visualization.py    # Utilitas visualisasi
│       └── metrics.py          # Perhitungan metrik kinerja
├── scripts/
│   ├── run_app.py              # UI berbasis GUI
│   ├── run_cli.py              # Interface command line
│   └── test_hedge_fund.py      # Script pengujian
├── results/                    # Folder hasil pengujian dan visualisasi
├── requirements.txt            # Dependensi
└── README.md                   # Dokumentasi
```

## 📝 Lisensi

[MIT](LICENSE)

## 🤝 Kontributor

- Anda! Kontribusi selalu disambut.

## 🔄 Future Work

- Integrasi dengan API broker 
- Penggunaan data real-time
- Optimasi hyperparameter otomatis
- Integrasi strategi tambahan
- Analisis sentimen dari berita keuangan 