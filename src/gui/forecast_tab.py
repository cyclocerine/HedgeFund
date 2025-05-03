"""
Forecast Tab Module
====================

Modul ini berisi implementasi tab trading prediksi untuk UI aplikasi.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QProgressBar, QDoubleSpinBox, QMessageBox,
                             QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout,
                             QSpinBox)
from PyQt5.QtCore import Qt, pyqtSlot, QDate
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..utils.worker_threads import ForecastTradeWorker

class ForecastTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = None
        self.forecast_prices = None
        self.setup_ui()
        
    def setup_ui(self):
        # Layout utama
        layout = QVBoxLayout(self)
        
        # Input parameters
        input_layout = QHBoxLayout()
        
        # Strategy selection
        strategy_layout = QVBoxLayout()
        strategy_label = QLabel("Trading Strategy:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['trend_following', 'mean_reversion', 'predictive', 'ppo'])
        self.strategy_combo.currentIndexChanged.connect(self.on_strategy_changed)
        strategy_layout.addWidget(strategy_label)
        strategy_layout.addWidget(self.strategy_combo)
        input_layout.addLayout(strategy_layout)
        
        # Initial investment
        investment_layout = QVBoxLayout()
        investment_label = QLabel("Initial Investment:")
        self.investment_spin = QDoubleSpinBox()
        self.investment_spin.setRange(1000, 1000000)
        self.investment_spin.setValue(10000)
        self.investment_spin.setSingleStep(1000)
        self.investment_spin.setPrefix("$ ")
        investment_layout.addWidget(investment_label)
        investment_layout.addWidget(self.investment_spin)
        input_layout.addLayout(investment_layout)
        
        layout.addLayout(input_layout)
        
        # Strategy parameters
        params_section = QHBoxLayout()
        
        # Trend Following parameters
        self.trend_following_group = QGroupBox("Trend Following Parameters")
        trend_following_layout = QFormLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.001, 0.1)
        self.threshold_spin.setSingleStep(0.001)
        self.threshold_spin.setValue(0.01)
        trend_following_layout.addRow("Threshold:", self.threshold_spin)
        self.trend_following_group.setLayout(trend_following_layout)
        params_section.addWidget(self.trend_following_group)
        
        # Mean Reversion parameters
        self.mean_reversion_group = QGroupBox("Mean Reversion Parameters")
        mean_reversion_layout = QFormLayout()
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 100)
        self.window_spin.setValue(20)
        self.buy_threshold_spin = QDoubleSpinBox()
        self.buy_threshold_spin.setRange(0.01, 0.2)
        self.buy_threshold_spin.setSingleStep(0.01)
        self.buy_threshold_spin.setValue(0.05)
        self.sell_threshold_spin = QDoubleSpinBox()
        self.sell_threshold_spin.setRange(0.01, 0.2)
        self.sell_threshold_spin.setSingleStep(0.01)
        self.sell_threshold_spin.setValue(0.05)
        mean_reversion_layout.addRow("Window Size:", self.window_spin)
        mean_reversion_layout.addRow("Buy Threshold:", self.buy_threshold_spin)
        mean_reversion_layout.addRow("Sell Threshold:", self.sell_threshold_spin)
        self.mean_reversion_group.setLayout(mean_reversion_layout)
        params_section.addWidget(self.mean_reversion_group)
        
        # Predictive parameters
        self.predictive_group = QGroupBox("Predictive Parameters")
        predictive_layout = QFormLayout()
        self.buy_multiplier_spin = QDoubleSpinBox()
        self.buy_multiplier_spin.setRange(1.001, 1.1)
        self.buy_multiplier_spin.setSingleStep(0.001)
        self.buy_multiplier_spin.setValue(1.01)
        self.sell_multiplier_spin = QDoubleSpinBox()
        self.sell_multiplier_spin.setRange(0.9, 0.999)
        self.sell_multiplier_spin.setSingleStep(0.001)
        self.sell_multiplier_spin.setValue(0.99)
        predictive_layout.addRow("Buy Threshold:", self.buy_multiplier_spin)
        predictive_layout.addRow("Sell Threshold:", self.sell_multiplier_spin)
        self.predictive_group.setLayout(predictive_layout)
        params_section.addWidget(self.predictive_group)
        
        # PPO parameters
        self.ppo_group = QGroupBox("PPO Parameters")
        ppo_layout = QFormLayout()
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(5, 100)
        self.episodes_spin.setValue(10)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.8, 0.999)
        self.gamma_spin.setSingleStep(0.001)
        self.gamma_spin.setValue(0.99)
        self.lam_spin = QDoubleSpinBox()
        self.lam_spin.setRange(0.8, 0.999)
        self.lam_spin.setSingleStep(0.001)
        self.lam_spin.setValue(0.95)
        ppo_layout.addRow("Episodes:", self.episodes_spin)
        ppo_layout.addRow("Gamma:", self.gamma_spin)
        ppo_layout.addRow("Lambda:", self.lam_spin)
        self.ppo_group.setLayout(ppo_layout)
        params_section.addWidget(self.ppo_group)
        
        # Show only relevant parameter group
        self.mean_reversion_group.hide()
        self.predictive_group.hide()
        self.ppo_group.hide()
        
        layout.addLayout(params_section)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Run button
        self.run_button = QPushButton("Simulate Trading")
        self.run_button.clicked.connect(self.run_forecast_trading)
        self.run_button.setEnabled(False)  # Disabled until prediction is done
        layout.addWidget(self.run_button)
        
        # Plot area
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Trade history table
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(5)
        self.trades_table.setHorizontalHeaderLabels(["Day", "Type", "Price", "Shares", "Value"])
        layout.addWidget(self.trades_table)
        
        # Performance metrics
        self.performance_label = QLabel("Performance Metrics:")
        layout.addWidget(self.performance_label)
        
        # Save button
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self.save_forecast_results)
        self.save_results_button.setEnabled(False)
        layout.addWidget(self.save_results_button)
    
    @pyqtSlot(object, object)
    def set_forecast_results(self, predictor, forecast):
        """
        Set the forecast results from the prediction tab
        
        Parameters:
        -----------
        predictor : StockPredictor
            The predictor object
        forecast : array-like
            Forecast prices from the model
        """
        self.predictor = predictor
        self.forecast_prices = forecast
        self.run_button.setEnabled(True)
    
    def on_strategy_changed(self):
        # Show relevant parameter group based on selected strategy
        strategy = self.strategy_combo.currentText()
        
        self.trend_following_group.hide()
        self.mean_reversion_group.hide()
        self.predictive_group.hide()
        self.ppo_group.hide()
        
        if strategy == 'trend_following':
            self.trend_following_group.show()
        elif strategy == 'mean_reversion':
            self.mean_reversion_group.show()
        elif strategy == 'predictive':
            self.predictive_group.show()
        elif strategy == 'ppo':
            self.ppo_group.show()
    
    def run_forecast_trading(self):
        # Disable run button
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        if self.forecast_prices is None:
            QMessageBox.warning(self, "Warning", "Please run prediction first")
            self.run_button.setEnabled(True)
            return
        
        # Get parameters
        strategy = self.strategy_combo.currentText()
        initial_investment = self.investment_spin.value()
        
        # Get strategy parameters
        params = {}
        if strategy == 'trend_following':
            params = {'threshold': self.threshold_spin.value()}
        elif strategy == 'mean_reversion':
            params = {
                'window': self.window_spin.value(),
                'buy_threshold': self.buy_threshold_spin.value(),
                'sell_threshold': self.sell_threshold_spin.value()
            }
        elif strategy == 'predictive':
            params = {
                'buy_threshold': self.buy_multiplier_spin.value(),
                'sell_threshold': self.sell_multiplier_spin.value()
            }
        elif strategy == 'ppo':
            params = {
                'episodes': self.episodes_spin.value(),
                'gamma': self.gamma_spin.value(),
                'lambda': self.lam_spin.value()
            }
        
        # Create and start worker thread
        self.worker = ForecastTradeWorker(self.forecast_prices, initial_investment, strategy, params)
        self.worker.finished.connect(self.on_forecast_trading_finished)
        self.worker.progress.connect(self.update_forecast_progress)
        self.worker.error.connect(self.show_forecast_error)
        self.worker.start()
    
    def update_forecast_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_forecast_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.run_button.setEnabled(True)
    
    def on_forecast_trading_finished(self, portfolio_values, trades, performance):
        # Update plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(portfolio_values, label='Portfolio Value')
        ax.set_title('Forecast Trading Simulation')
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True)
        self.canvas.draw()
        
        # Update trades table
        self.trades_table.setRowCount(len(trades))
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade['day'])))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade['type']))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"${trade['price']:.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"{trade['shares']:.4f}"))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"${trade['value']:.2f}"))
        
        # Update performance metrics
        performance_text = ""
        for key, value in performance.items():
            if key in ['total_return', 'max_drawdown', 'win_rate']:
                performance_text += f"{key}: {value:.2f}%\n"
            elif isinstance(value, float):
                performance_text += f"{key}: ${value:.2f}\n"
            else:
                performance_text += f"{key}: {value}\n"
        self.performance_label.setText(f"Performance Metrics:\n{performance_text}")
        
        # Enable buttons
        self.run_button.setEnabled(True)
        self.save_results_button.setEnabled(True)
    
    def save_forecast_results(self):
        # Implement save results functionality
        pass 