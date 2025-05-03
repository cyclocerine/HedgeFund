"""
GUI Application
=============

Modul ini berisi kelas utama aplikasi GUI.
"""

from PyQt5.QtWidgets import QMainWindow, QTabWidget, QApplication, QStyleFactory, QWidget, QVBoxLayout, QLabel, QAction, QMenu
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont, QPixmap

from .prediction_tab import PredictionTab
from .backtest_tab import BacktestTab
from .optimizer_tab import OptimizerTab
from .forecast_tab import ForecastTab
from .styles import MAIN_STYLESHEET, DARK_STYLESHEET

class StockPredictionApp(QMainWindow):
    def __init__(self, use_dark_theme=False, style_name='fusion'):
        """
        Inisialisasi aplikasi utama
        
        Parameters:
        -----------
        use_dark_theme : bool
            Flag untuk menggunakan tema gelap
        style_name : str
            Nama style yang digunakan (fusion, windows, dll)
        """
        super().__init__()
        self.setWindowTitle("Stock Price Prediction - Analysis Tool")
        self.setGeometry(100, 100, 1280, 900)
        
        # Simpan pengaturan tema
        self.use_dark_theme = use_dark_theme
        self.style_name = style_name
        
        # Atur tema dan style
        self._setup_style()
        
        # Widget utama
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        self.setCentralWidget(main_widget)
        
        # Header dengan logo
        header_widget = self._create_header()
        main_layout.addWidget(header_widget)
        
        # Tab widget untuk memisahkan prediksi, backtesting, dan optimizer
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(True)
        main_layout.addWidget(self.tabs)
        
        # Tab untuk prediksi
        self.prediction_tab = PredictionTab()
        self.tabs.addTab(self.prediction_tab, "Prediksi")
        
        # Tab untuk backtesting
        self.backtest_tab = BacktestTab()
        self.tabs.addTab(self.backtest_tab, "Backtesting")
        
        # Tab untuk optimizer
        self.optimizer_tab = OptimizerTab()
        self.tabs.addTab(self.optimizer_tab, "Optimizer Strategi")
        
        # Tab untuk simulasi trading pada forecast
        self.forecast_tab = ForecastTab()
        self.tabs.addTab(self.forecast_tab, "Trading Prediksi")
        
        # Hubungkan tab prediction dengan backtest dan forecast
        self.prediction_tab.prediction_finished.connect(self._handle_prediction_finished)
        
        # Set status bar
        self.statusBar().showMessage("Siap untuk melakukan analisis saham")
        
        # Buat menu
        self._create_menus()
        
    def _create_header(self):
        """Membuat header aplikasi dengan logo dan judul"""
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        title_label = QLabel("Stock Price Prediction")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        
        if self.use_dark_theme:
            title_label.setStyleSheet("color: #ecf0f1;")
        else:
            title_label.setStyleSheet("color: #2c3e50;")
        
        subtitle_label = QLabel("Analisis dan Prediksi Harga Saham dengan Deep Learning")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("Arial", 10))
        
        if self.use_dark_theme:
            subtitle_label.setStyleSheet("color: #bdc3c7;")
        else:
            subtitle_label.setStyleSheet("color: #7f8c8d;")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        
        return header_widget
        
    def _setup_style(self):
        """Mengatur style aplikasi"""
        QApplication.setStyle(QStyleFactory.create(self.style_name))
    
    def _create_menus(self):
        """Membuat menu aplikasi"""
        # Menu File
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        
        # Action Export
        export_action = QAction("Export Hasil", self)
        export_action.setStatusTip("Export hasil analisis")
        file_menu.addAction(export_action)
        
        # Action Exit
        exit_action = QAction("Keluar", self)
        exit_action.setStatusTip("Keluar aplikasi")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Tools
        tools_menu = menu_bar.addMenu("Tools")
        
        # Action untuk download data
        download_action = QAction("Download Data", self)
        download_action.setStatusTip("Download data saham dari sumber eksternal")
        tools_menu.addAction(download_action)
        
        # Menu bantuan
        help_menu = menu_bar.addMenu("Bantuan")
        
        # Action Dokumentasi
        docs_action = QAction("Dokumentasi", self)
        docs_action.setStatusTip("Buka dokumentasi aplikasi")
        help_menu.addAction(docs_action)
        
        # Action About
        about_action = QAction("Tentang", self)
        about_action.setStatusTip("Informasi tentang aplikasi")
        help_menu.addAction(about_action)
        
    def _handle_prediction_finished(self, predictor, y_true, y_pred, forecast):
        """
        Menangani event setelah prediksi selesai
        
        Parameters:
        -----------
        predictor : StockPredictor
            Objek prediktor yang menghasilkan prediksi
        y_true : array-like
            Array harga aktual historis
        y_pred : array-like
            Array harga prediksi dari model
        forecast : array-like
            Array harga forecast ke depan
        """
        # Kirim hasil ke tab backtest
        self.backtest_tab.set_prediction_results(predictor, y_true, y_pred)
        
        # Kirim hasil ke tab forecast
        self.forecast_tab.set_forecast_results(predictor, forecast)
        
        # Update status bar
        self.statusBar().showMessage(f"Prediksi selesai untuk {predictor.ticker} menggunakan model {predictor.model_type}") 