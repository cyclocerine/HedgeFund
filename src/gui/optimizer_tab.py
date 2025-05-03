"""
Optimizer Tab Module
====================

Modul ini berisi implementasi tab optimizer strategi untuk UI aplikasi.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QProgressBar, QDoubleSpinBox, QMessageBox,
                             QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout, 
                             QSpinBox, QDateEdit)
from PyQt5.QtCore import Qt, QDate
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..utils.worker_threads import StrategyOptimizerWorker

class OptimizerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # Layout utama
        layout = QVBoxLayout(self)
        
        # Input parameters
        input_layout = QHBoxLayout()
        
        # Ticker input
        ticker_layout = QVBoxLayout()
        ticker_label = QLabel("Ticker Symbol:")
        self.ticker_input = QComboBox()
        self.ticker_input.setEditable(True)
        self.ticker_input.addItems(['ADRO.JK', 'BBCA.JK', 'BBRI.JK', 'TLKM.JK', 'ASII.JK'])
        ticker_layout.addWidget(ticker_label)
        ticker_layout.addWidget(self.ticker_input)
        input_layout.addLayout(ticker_layout)
        
        # Date inputs
        date_layout = QVBoxLayout()
        start_date_label = QLabel("Start Date:")
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate(2020, 1, 1))
        end_date_label = QLabel("End Date:")
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        date_layout.addWidget(start_date_label)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(end_date_label)
        date_layout.addWidget(self.end_date)
        input_layout.addLayout(date_layout)
        
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
        
        # Strategy parameters groups
        param_section = QHBoxLayout()
        
        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(['cnn_lstm', 'bilstm', 'transformer', 'ensemble'])
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        param_section.addWidget(model_group)
        
        # Strategy selection
        strategy_group = QGroupBox("Strategy")
        strategy_layout = QVBoxLayout()
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['trend_following', 'mean_reversion', 'predictive', 'PPO'])
        self.strategy_combo.currentIndexChanged.connect(self.on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo)
        strategy_group.setLayout(strategy_layout)
        param_section.addWidget(strategy_group)
        
        # Parameter ranges
        self.trend_following_group = QGroupBox("Trend Following Parameters")
        trend_following_layout = QFormLayout()
        self.trend_thresholds = []
        threshold_min = QDoubleSpinBox()
        threshold_min.setRange(0.001, 0.1)
        threshold_min.setSingleStep(0.001)
        threshold_min.setValue(0.005)
        threshold_max = QDoubleSpinBox()
        threshold_max.setRange(0.001, 0.1)
        threshold_max.setSingleStep(0.001)
        threshold_max.setValue(0.02)
        threshold_step = QDoubleSpinBox()
        threshold_step.setRange(0.001, 0.01)
        threshold_step.setSingleStep(0.001)
        threshold_step.setValue(0.005)
        trend_following_layout.addRow("Min Threshold:", threshold_min)
        trend_following_layout.addRow("Max Threshold:", threshold_max)
        trend_following_layout.addRow("Step:", threshold_step)
        self.trend_thresholds = [threshold_min, threshold_max, threshold_step]
        self.trend_following_group.setLayout(trend_following_layout)
        param_section.addWidget(self.trend_following_group)
        
        self.mean_reversion_group = QGroupBox("Mean Reversion Parameters")
        mean_reversion_layout = QFormLayout()
        self.window_sizes = []
        window_min = QSpinBox()
        window_min.setRange(5, 50)
        window_min.setValue(10)
        window_max = QSpinBox()
        window_max.setRange(10, 100)
        window_max.setValue(30)
        window_step = QSpinBox()
        window_step.setRange(1, 10)
        window_step.setValue(5)
        
        self.buy_thresholds = []
        buy_threshold_min = QDoubleSpinBox()
        buy_threshold_min.setRange(0.01, 0.2)
        buy_threshold_min.setSingleStep(0.01)
        buy_threshold_min.setValue(0.05)
        buy_threshold_max = QDoubleSpinBox()
        buy_threshold_max.setRange(0.01, 0.2)
        buy_threshold_max.setSingleStep(0.01)
        buy_threshold_max.setValue(0.1)
        buy_threshold_step = QDoubleSpinBox()
        buy_threshold_step.setRange(0.01, 0.05)
        buy_threshold_step.setSingleStep(0.01)
        buy_threshold_step.setValue(0.01)
        
        self.sell_thresholds = []
        sell_threshold_min = QDoubleSpinBox()
        sell_threshold_min.setRange(0.01, 0.2)
        sell_threshold_min.setSingleStep(0.01)
        sell_threshold_min.setValue(0.05)
        sell_threshold_max = QDoubleSpinBox()
        sell_threshold_max.setRange(0.01, 0.2)
        sell_threshold_max.setSingleStep(0.01)
        sell_threshold_max.setValue(0.1)
        sell_threshold_step = QDoubleSpinBox()
        sell_threshold_step.setRange(0.01, 0.05)
        sell_threshold_step.setSingleStep(0.01)
        sell_threshold_step.setValue(0.01)
        
        mean_reversion_layout.addRow("Min Window Size:", window_min)
        mean_reversion_layout.addRow("Max Window Size:", window_max)
        mean_reversion_layout.addRow("Window Step:", window_step)
        mean_reversion_layout.addRow("Min Buy Threshold:", buy_threshold_min)
        mean_reversion_layout.addRow("Max Buy Threshold:", buy_threshold_max)
        mean_reversion_layout.addRow("Buy Threshold Step:", buy_threshold_step)
        mean_reversion_layout.addRow("Min Sell Threshold:", sell_threshold_min)
        mean_reversion_layout.addRow("Max Sell Threshold:", sell_threshold_max)
        mean_reversion_layout.addRow("Sell Threshold Step:", sell_threshold_step)
        
        self.window_sizes = [window_min, window_max, window_step]
        self.buy_thresholds = [buy_threshold_min, buy_threshold_max, buy_threshold_step]
        self.sell_thresholds = [sell_threshold_min, sell_threshold_max, sell_threshold_step]
        
        self.mean_reversion_group.setLayout(mean_reversion_layout)
        param_section.addWidget(self.mean_reversion_group)
        
        self.predictive_group = QGroupBox("Predictive Parameters")
        predictive_layout = QFormLayout()
        self.buy_multipliers = []
        buy_mult_min = QDoubleSpinBox()
        buy_mult_min.setRange(1.001, 1.05)
        buy_mult_min.setSingleStep(0.001)
        buy_mult_min.setValue(1.005)
        buy_mult_max = QDoubleSpinBox()
        buy_mult_max.setRange(1.001, 1.05)
        buy_mult_max.setSingleStep(0.001)
        buy_mult_max.setValue(1.02)
        buy_mult_step = QDoubleSpinBox()
        buy_mult_step.setRange(0.001, 0.01)
        buy_mult_step.setSingleStep(0.001)
        buy_mult_step.setValue(0.005)
        
        self.sell_multipliers = []
        sell_mult_min = QDoubleSpinBox()
        sell_mult_min.setRange(0.9, 0.999)
        sell_mult_min.setSingleStep(0.001)
        sell_mult_min.setValue(0.98)
        sell_mult_max = QDoubleSpinBox()
        sell_mult_max.setRange(0.9, 0.999)
        sell_mult_max.setSingleStep(0.001)
        sell_mult_max.setValue(0.995)
        sell_mult_step = QDoubleSpinBox()
        sell_mult_step.setRange(0.001, 0.01)
        sell_mult_step.setSingleStep(0.001)
        sell_mult_step.setValue(0.005)
        
        predictive_layout.addRow("Min Buy Multiplier:", buy_mult_min)
        predictive_layout.addRow("Max Buy Multiplier:", buy_mult_max)
        predictive_layout.addRow("Buy Multiplier Step:", buy_mult_step)
        predictive_layout.addRow("Min Sell Multiplier:", sell_mult_min)
        predictive_layout.addRow("Max Sell Multiplier:", sell_mult_max)
        predictive_layout.addRow("Sell Multiplier Step:", sell_mult_step)
        
        self.buy_multipliers = [buy_mult_min, buy_mult_max, buy_mult_step]
        self.sell_multipliers = [sell_mult_min, sell_mult_max, sell_mult_step]
        
        self.predictive_group.setLayout(predictive_layout)
        param_section.addWidget(self.predictive_group)
        
        # PPO Parameters
        self.ppo_group = QGroupBox("PPO Parameters")
        ppo_layout = QFormLayout()
        
        self.episodes_range = []
        episodes_min = QSpinBox()
        episodes_min.setRange(5, 50)
        episodes_min.setValue(5)
        episodes_max = QSpinBox()
        episodes_max.setRange(10, 100)
        episodes_max.setValue(20)
        episodes_step = QSpinBox()
        episodes_step.setRange(5, 10)
        episodes_step.setValue(5)
        
        self.gamma_range = []
        gamma_min = QDoubleSpinBox()
        gamma_min.setRange(0.9, 0.99)
        gamma_min.setSingleStep(0.01)
        gamma_min.setValue(0.95)
        gamma_max = QDoubleSpinBox()
        gamma_max.setRange(0.9, 0.999)
        gamma_max.setSingleStep(0.01)
        gamma_max.setValue(0.99)
        gamma_step = QDoubleSpinBox()
        gamma_step.setRange(0.01, 0.05)
        gamma_step.setSingleStep(0.01)
        gamma_step.setValue(0.01)
        
        self.lambda_range = []
        lambda_min = QDoubleSpinBox()
        lambda_min.setRange(0.9, 0.99)
        lambda_min.setSingleStep(0.01)
        lambda_min.setValue(0.9)
        lambda_max = QDoubleSpinBox()
        lambda_max.setRange(0.9, 0.99)
        lambda_max.setSingleStep(0.01)
        lambda_max.setValue(0.99)
        lambda_step = QDoubleSpinBox()
        lambda_step.setRange(0.01, 0.05)
        lambda_step.setSingleStep(0.01)
        lambda_step.setValue(0.03)
        
        ppo_layout.addRow("Min Episodes:", episodes_min)
        ppo_layout.addRow("Max Episodes:", episodes_max)
        ppo_layout.addRow("Episodes Step:", episodes_step)
        ppo_layout.addRow("Min Gamma:", gamma_min)
        ppo_layout.addRow("Max Gamma:", gamma_max)
        ppo_layout.addRow("Gamma Step:", gamma_step)
        ppo_layout.addRow("Min Lambda:", lambda_min)
        ppo_layout.addRow("Max Lambda:", lambda_max)
        ppo_layout.addRow("Lambda Step:", lambda_step)
        
        self.episodes_range = [episodes_min, episodes_max, episodes_step]
        self.gamma_range = [gamma_min, gamma_max, gamma_step]
        self.lambda_range = [lambda_min, lambda_max, lambda_step]
        
        self.ppo_group.setLayout(ppo_layout)
        param_section.addWidget(self.ppo_group)
        
        # Show only relevant parameter group
        self.mean_reversion_group.hide()
        self.predictive_group.hide()
        self.ppo_group.hide()
        
        layout.addLayout(param_section)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Run button
        self.run_button = QPushButton("Run Optimizer")
        self.run_button.clicked.connect(self.run_strategy_optimization)
        layout.addWidget(self.run_button)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels(
            ["Strategy", "Parameters", "Initial", "Final", "Return (%)", 
             "Max Drawdown (%)", "Win Rate (%)", "Trades"]
        )
        layout.addWidget(self.results_table)
        
        # Save button
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self.save_optimization_results)
        self.save_results_button.setEnabled(False)
        layout.addWidget(self.save_results_button)
    
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
        elif strategy in ['PPO', 'ppo']:
            self.ppo_group.show()
    
    def run_strategy_optimization(self):
        # Disable run button
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Get parameters
        ticker = self.ticker_input.currentText()
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        model = self.model_combo.currentText()
        strategy = self.strategy_combo.currentText()
        initial_investment = self.investment_spin.value()
        
        # Generate parameter ranges based on strategy
        param_ranges = {}
        
        if strategy == 'trend_following':
            threshold_min, threshold_max, threshold_step = [s.value() for s in self.trend_thresholds]
            thresholds = []
            current = threshold_min
            while current <= threshold_max:
                thresholds.append(round(current, 3))
                current += threshold_step
                
            param_ranges = {'threshold': thresholds}
            
        elif strategy == 'mean_reversion':
            window_min, window_max, window_step = [s.value() for s in self.window_sizes]
            buy_min, buy_max, buy_step = [s.value() for s in self.buy_thresholds]
            sell_min, sell_max, sell_step = [s.value() for s in self.sell_thresholds]
            
            windows = []
            current = window_min
            while current <= window_max:
                windows.append(int(current))
                current += window_step
                
            buy_thresholds = []
            current = buy_min
            while current <= buy_max:
                buy_thresholds.append(round(current, 3))
                current += buy_step
                
            sell_thresholds = []
            current = sell_min
            while current <= sell_max:
                sell_thresholds.append(round(current, 3))
                current += sell_step
                
            param_ranges = {
                'window': windows,
                'buy_threshold': buy_thresholds,
                'sell_threshold': sell_thresholds
            }
            
        elif strategy == 'predictive':
            buy_min, buy_max, buy_step = [s.value() for s in self.buy_multipliers]
            sell_min, sell_max, sell_step = [s.value() for s in self.sell_multipliers]
            
            buy_multipliers = []
            current = buy_min
            while current <= buy_max:
                buy_multipliers.append(round(current, 3))
                current += buy_step
                
            sell_multipliers = []
            current = sell_min
            while current <= sell_max:
                sell_multipliers.append(round(current, 3))
                current += sell_step
                
            param_ranges = {
                'buy_threshold': buy_multipliers,
                'sell_threshold': sell_multipliers
            }
            
        elif strategy in ['PPO', 'ppo']:
            episodes_min, episodes_max, episodes_step = [s.value() for s in self.episodes_range]
            gamma_min, gamma_max, gamma_step = [s.value() for s in self.gamma_range]
            lambda_min, lambda_max, lambda_step = [s.value() for s in self.lambda_range]
            
            episodes_list = []
            current = episodes_min
            while current <= episodes_max:
                episodes_list.append(int(current))
                current += episodes_step
                
            param_ranges = {
                'episodes': episodes_list,
                'gamma': {
                    'min': gamma_min,
                    'max': gamma_max,
                    'step': gamma_step
                },
                'lambda': {
                    'min': lambda_min,
                    'max': lambda_max,
                    'step': lambda_step
                }
            }
        
        # Create worker thread
        self.worker = StrategyOptimizerWorker(
            ticker, start_date, end_date, model, initial_investment,
            strategy, param_ranges
        )
        
        # Connect signals
        self.worker.progress.connect(self.update_opt_progress)
        self.worker.finished.connect(self.on_optimization_finished)
        self.worker.error.connect(self.show_opt_error)
        
        # Start worker
        self.worker.start()
        
    def update_opt_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_opt_error(self, message):
        QMessageBox.critical(self, "Error", f"Error during optimization: {message}")
        self.run_button.setEnabled(True)
    
    def on_optimization_finished(self, results):
        # Update results table
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            strategy = QTableWidgetItem(result['strategy'])
            
            # Format parameters based on strategy
            params_text = ""
            if result['strategy'] == 'trend_following':
                params_text = f"Threshold: {result['params']['threshold']:.3f}"
            elif result['strategy'] == 'mean_reversion':
                params_text = f"Window: {result['params']['window']}, "
                params_text += f"Buy: {result['params']['buy_threshold']:.3f}, "
                params_text += f"Sell: {result['params']['sell_threshold']:.3f}"
            elif result['strategy'] == 'predictive':
                params_text = f"Buy: {result['params']['buy_threshold']:.3f}, "
                params_text += f"Sell: {result['params']['sell_threshold']:.3f}"
            elif result['strategy'] in ['PPO', 'ppo']:
                params_text = f"Episodes: {result['params']['episodes']}, "
                params_text += f"Gamma: {result['params']['gamma']:.3f}, "
                params_text += f"Lambda: {result['params']['lambda']:.2f}"
                
            params = QTableWidgetItem(params_text)
            
            initial = QTableWidgetItem(f"{result['initial_investment']:.2f}")
            final = QTableWidgetItem(f"{result['final_value']:.2f}")
            total_return = QTableWidgetItem(f"{result['total_return']:.2f}")
            drawdown = QTableWidgetItem(f"{result['max_drawdown']:.2f}")
            win_rate = QTableWidgetItem(f"{result['win_rate']:.2f}")
            trades = QTableWidgetItem(f"{result['num_trades']}")
            
            self.results_table.setItem(i, 0, strategy)
            self.results_table.setItem(i, 1, params)
            self.results_table.setItem(i, 2, initial)
            self.results_table.setItem(i, 3, final)
            self.results_table.setItem(i, 4, total_return)
            self.results_table.setItem(i, 5, drawdown)
            self.results_table.setItem(i, 6, win_rate)
            self.results_table.setItem(i, 7, trades)
        
        # Enable save button
        self.save_results_button.setEnabled(True)
        self.run_button.setEnabled(True)
    
    def save_optimization_results(self):
        # Implement save results functionality
        pass 