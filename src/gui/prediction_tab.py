"""
Prediction Tab Module
====================

Modul ini berisi implementasi tab prediksi untuk UI aplikasi.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                             QDateEdit, QComboBox, QSpinBox, QCheckBox, QPushButton, 
                             QProgressBar, QMessageBox, QFrame, QSplitter, QGroupBox,
                             QFileDialog, QToolButton, QSizePolicy)
from PyQt5.QtCore import QDate, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.predictor import StockPredictor
from ..utils.worker_threads import WorkerThread

class StyledGroupBox(QGroupBox):
    """Custom styled group box for better appearance"""
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

class ResultCard(QFrame):
    """Card widget untuk menampilkan hasil"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 5px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.title_label.setStyleSheet("color: #2c3e50;")
        self.layout.addWidget(self.title_label)
        
        self.content = QLabel()
        self.content.setWordWrap(True)
        self.layout.addWidget(self.content)
        
    def set_content(self, text):
        self.content.setText(text)

class PredictionTab(QWidget):
    # Signal untuk memberitahu tab lain bahwa prediksi selesai
    prediction_finished = pyqtSignal(object, object, object, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = None
        self.setup_ui()
        
    def setup_ui(self):
        # Widget utama
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Splitter untuk membagi form input dan hasil
        splitter = QSplitter(Qt.Vertical)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(splitter)
        
        # === Form Input ===
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(input_widget)
        
        # Group box untuk input parameters
        params_group = StyledGroupBox("Parameter Prediksi")
        input_layout.addWidget(params_group)
        
        params_layout = QHBoxLayout(params_group)
        params_layout.setSpacing(20)
        
        # Ticker input
        ticker_layout = QVBoxLayout()
        ticker_label = QLabel("Ticker Symbol:")
        self.ticker_input = QLineEdit("ADRO.JK")
        ticker_layout.addWidget(ticker_label)
        ticker_layout.addWidget(self.ticker_input)
        params_layout.addLayout(ticker_layout)
        
        # Date inputs
        date_layout = QVBoxLayout()
        start_date_label = QLabel("Start Date:")
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate(2020, 1, 1))
        self.start_date.setCalendarPopup(True)
        end_date_label = QLabel("End Date:")
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        date_layout.addWidget(start_date_label)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(end_date_label)
        date_layout.addWidget(self.end_date)
        params_layout.addLayout(date_layout)
        
        # Model selection
        model_layout = QVBoxLayout()
        model_label = QLabel("Model Type:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['cnn_lstm', 'bilstm', 'transformer', 'ensemble'])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        params_layout.addLayout(model_layout)
        
        # Parameters
        param_layout = QVBoxLayout()
        lookback_label = QLabel("Lookback Days:")
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(10, 200)
        self.lookback_spin.setValue(60)
        forecast_label = QLabel("Forecast Days:")
        self.forecast_spin = QSpinBox()
        self.forecast_spin.setRange(1, 100)
        self.forecast_spin.setValue(20)
        param_layout.addWidget(lookback_label)
        param_layout.addWidget(self.lookback_spin)
        param_layout.addWidget(forecast_label)
        param_layout.addWidget(self.forecast_spin)
        params_layout.addLayout(param_layout)
        
        # Hyperparameter tuning
        tune_layout = QVBoxLayout()
        tune_label = QLabel("Tuning:")
        self.tune_checkbox = QCheckBox("Enable Hyperparameter Tuning")
        tune_layout.addWidget(tune_label)
        tune_layout.addWidget(self.tune_checkbox)
        params_layout.addLayout(tune_layout)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        input_layout.addLayout(actions_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        actions_layout.addWidget(self.progress_bar, 3)
        
        # Run button
        self.run_button = QPushButton("Run Prediction")
        self.run_button.setIcon(QIcon.fromTheme("system-run"))
        self.run_button.clicked.connect(self.run_prediction)
        actions_layout.addWidget(self.run_button, 1)
        
        # === Results Section ===
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(results_widget)
        
        # Results splitter - split between plot and metrics
        results_splitter = QSplitter(Qt.Horizontal)
        results_layout.addWidget(results_splitter)
        
        # Plot area on left side
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        results_splitter.addWidget(plot_widget)
        
        plot_group = StyledGroupBox("Price Prediction Chart")
        plot_layout.addWidget(plot_group)
        
        plot_inner_layout = QVBoxLayout(plot_group)
        
        # Plot toolbar
        toolbar_layout = QHBoxLayout()
        plot_inner_layout.addLayout(toolbar_layout)
        
        self.save_plot_button = QToolButton()
        self.save_plot_button.setText("Save Plot")
        self.save_plot_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.save_plot_button.setIcon(QIcon.fromTheme("document-save"))
        self.save_plot_button.clicked.connect(self.save_plot)
        toolbar_layout.addWidget(self.save_plot_button)
        
        toolbar_layout.addStretch()
        
        # Plot canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.set_facecolor('#f9f9f9')
        self.canvas = FigureCanvas(self.figure)
        plot_inner_layout.addWidget(self.canvas)
        
        # Cards on right side
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        results_splitter.addWidget(cards_widget)
        
        # Metrics card
        self.metrics_card = ResultCard("Model Performance Metrics")
        cards_layout.addWidget(self.metrics_card)
        
        # Forecast card
        self.forecast_card = ResultCard("Forecast Results")
        cards_layout.addWidget(self.forecast_card)
        
        # Save results button
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.setIcon(QIcon.fromTheme("document-save"))
        self.save_results_button.clicked.connect(self.save_results)
        cards_layout.addWidget(self.save_results_button)
        
        # Set initial splitter sizes - 30% for input, 70% for results
        splitter.setSizes([300, 700])
        results_splitter.setSizes([600, 400])
        
        # Disable buttons initially
        self.save_plot_button.setEnabled(False)
        self.save_results_button.setEnabled(False)
        
        # Set tab order
        self.setTabOrder(self.ticker_input, self.start_date)
        self.setTabOrder(self.start_date, self.end_date)
        self.setTabOrder(self.end_date, self.model_combo)
        self.setTabOrder(self.model_combo, self.lookback_spin)
        self.setTabOrder(self.lookback_spin, self.forecast_spin)
        self.setTabOrder(self.forecast_spin, self.tune_checkbox)
        self.setTabOrder(self.tune_checkbox, self.run_button)
    
    def run_prediction(self):
        # Disable run button during prediction
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Get parameters
        ticker = self.ticker_input.text()
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        model_type = self.model_combo.currentText()
        lookback = self.lookback_spin.value()
        forecast_days = self.forecast_spin.value()
        tune = self.tune_checkbox.isChecked()
        
        # Create predictor
        self.predictor = StockPredictor(
            ticker, start_date, end_date,
            lookback, forecast_days, model_type, tune
        )
        
        # Create and start worker thread
        self.worker = WorkerThread(self.predictor)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.run_button.setEnabled(True)
    
    def on_prediction_finished(self, y_true, y_pred, forecast, metrics):
        # Update plot
        self.figure.clear()
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        ax = self.figure.add_subplot(111)
        
        # Plot actual values
        ax.plot(y_true, label='Aktual', color='#3498db', linewidth=2)
        
        # Plot predicted values
        ax.plot(y_pred, label='Prediksi', color='#e74c3c', linewidth=2, linestyle='--')
        
        # Plot forecast values
        ax.plot(range(len(y_true), len(y_true) + len(forecast)), 
                forecast, label='Forecast', color='#2ecc71', linewidth=2.5)
        
        # Fill area under forecast
        forecast_range = range(len(y_true), len(y_true) + len(forecast))
        min_val = min(min(y_true), min(y_pred), min(forecast)) * 0.95
        ax.fill_between(forecast_range, min_val, forecast, alpha=0.2, color='#2ecc71')
        
        # Set title and style
        ax.set_title(f'{self.predictor.ticker} Stock Price Prediction', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Harga', fontsize=12)
        
        # Style the grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend with better style
        ax.legend(loc='best', frameon=True, shadow=True)
        
        # Tight layout for better spacing
        self.figure.tight_layout()
        
        # Redraw canvas
        self.canvas.draw()
        
        # Update metrics
        metrics_str = ""
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_str += f"<b>{key}</b>: {value:.4f}<br>"
            else:
                metrics_str += f"<b>{key}</b>: {value}<br>"
        self.metrics_card.set_content(metrics_str)
        
        # Update forecast
        forecast_str = "<b>Prediksi harga untuk hari-hari berikutnya:</b><br><br>"
        for i, price in enumerate(forecast, 1):
            forecast_str += f"Hari ke-{i}: {price:.2f}<br>"
            
            # Add trend indicator
            if i > 1:
                prev_price = forecast[i-2]
                pct_change = (price - prev_price) / prev_price * 100
                if pct_change > 0:
                    forecast_str += f"<span style='color:#2ecc71'>(+{pct_change:.2f}%)</span><br>"
                else:
                    forecast_str += f"<span style='color:#e74c3c'>({pct_change:.2f}%)</span><br>"
        
        self.forecast_card.set_content(forecast_str)
        
        # Enable buttons
        self.save_plot_button.setEnabled(True)
        self.save_results_button.setEnabled(True)
        self.run_button.setEnabled(True)
        
        # Emit signal to other tabs
        self.prediction_finished.emit(self.predictor, y_true, y_pred, forecast)
    
    def save_plot(self):
        # Implementasi penyimpanan plot
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Plot", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf);;SVG (*.svg)"
        )
        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Sukses", f"Plot berhasil disimpan ke {file_path}")
    
    def save_results(self):
        # Implementasi penyimpanan hasil
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Hasil", "", "CSV (*.csv);;Excel (*.xlsx)"
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.predictor.save_results_to_csv(file_path)
                else:
                    self.predictor.save_results_to_excel(file_path)
                QMessageBox.information(self, "Sukses", f"Hasil berhasil disimpan ke {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menyimpan hasil: {str(e)}") 