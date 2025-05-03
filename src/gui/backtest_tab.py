"""
Backtest Tab Module
====================

Modul ini berisi implementasi tab backtesting untuk UI aplikasi.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QProgressBar, QDoubleSpinBox, QMessageBox,
                             QTableWidget, QTableWidgetItem, QFrame, QSplitter, QGroupBox,
                             QHeaderView, QFileDialog, QSpinBox, QToolButton, QSizePolicy,
                             QCheckBox, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.QtGui import QColor, QBrush, QFont, QIcon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from ..utils.worker_threads import BacktestWorker
from .prediction_tab import StyledGroupBox, ResultCard

class BacktestTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = None
        self.actual_prices = None
        self.predicted_prices = None
        self.backtest_result = None
        self.setup_ui()
        
    def setup_ui(self):
        # Layout utama
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Splitter utama untuk membagi parameter/controls dan hasil
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(main_splitter)
        
        # === Panel Kontrol ===
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter.addWidget(control_widget)
        
        # Group box untuk parameter
        params_group = StyledGroupBox("Parameter Backtesting")
        control_layout.addWidget(params_group)
        
        # Ubah layout ke grid untuk parameter yang lebih banyak
        params_layout = QGridLayout(params_group)
        params_layout.setSpacing(10)
        
        # Kolom 1 - Parameter strategi
        col = 0
        row = 0
        
        # Strategy selection
        strategy_label = QLabel("Strategi Trading:")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['trend_following', 'mean_reversion', 'predictive', 'PPO'])
        params_layout.addWidget(strategy_label, row, col)
        params_layout.addWidget(self.strategy_combo, row+1, col)
        
        # Initial investment
        row += 2
        investment_label = QLabel("Modal Awal:")
        self.investment_spin = QDoubleSpinBox()
        self.investment_spin.setRange(1000, 10000000)
        self.investment_spin.setValue(10000)
        self.investment_spin.setSingleStep(1000)
        self.investment_spin.setPrefix("Rp ")
        self.investment_spin.setGroupSeparatorShown(True)
        params_layout.addWidget(investment_label, row, col)
        params_layout.addWidget(self.investment_spin, row+1, col)
        
        # Commission fee
        row += 2
        commission_label = QLabel("Komisi (%):")
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 2)
        self.commission_spin.setValue(0.15)
        self.commission_spin.setSingleStep(0.05)
        self.commission_spin.setSuffix(" %")
        params_layout.addWidget(commission_label, row, col)
        params_layout.addWidget(self.commission_spin, row+1, col)
                
        # Kolom 2 - Parameter risk management
        col = 1
        row = 0
        
        # Risk - Max Position Size
        position_size_label = QLabel("Ukuran Posisi Maks (%):")
        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(0.1, 100)
        self.position_size_spin.setValue(20)
        self.position_size_spin.setSingleStep(5)
        self.position_size_spin.setSuffix(" %")
        params_layout.addWidget(position_size_label, row, col)
        params_layout.addWidget(self.position_size_spin, row+1, col)
        
        # Stop Loss
        row += 2
        stop_loss_label = QLabel("Stop Loss (%):")
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.1, 20)
        self.stop_loss_spin.setValue(5)
        self.stop_loss_spin.setSingleStep(0.5)
        self.stop_loss_spin.setSuffix(" %")
        params_layout.addWidget(stop_loss_label, row, col)
        params_layout.addWidget(self.stop_loss_spin, row+1, col)
        
        # Trailing Stop
        row += 2
        trailing_stop_label = QLabel("Trailing Stop (%):")
        self.trailing_stop_spin = QDoubleSpinBox()
        self.trailing_stop_spin.setRange(0.1, 20)
        self.trailing_stop_spin.setValue(3)
        self.trailing_stop_spin.setSingleStep(0.5)
        self.trailing_stop_spin.setSuffix(" %")
        params_layout.addWidget(trailing_stop_label, row, col)
        params_layout.addWidget(self.trailing_stop_spin, row+1, col)
        
        # Kolom 3 - Parameter tambahan
        col = 2
        row = 0
        
        # Short selling enabled
        short_selling_label = QLabel("Short Selling:")
        self.short_selling_check = QCheckBox("Aktifkan")
        self.short_selling_check.setChecked(False)
        params_layout.addWidget(short_selling_label, row, col)
        params_layout.addWidget(self.short_selling_check, row+1, col)
        
        # Max Drawdown
        row += 2
        max_drawdown_label = QLabel("Max Drawdown (%):")
        self.max_drawdown_spin = QDoubleSpinBox()
        self.max_drawdown_spin.setRange(1, 50)
        self.max_drawdown_spin.setValue(10)
        self.max_drawdown_spin.setSingleStep(1)
        self.max_drawdown_spin.setSuffix(" %")
        params_layout.addWidget(max_drawdown_label, row, col)
        params_layout.addWidget(self.max_drawdown_spin, row+1, col)
        
        # Max Capital per Trade
        row += 2
        max_capital_label = QLabel("Kapital per Trade (%):")
        self.max_capital_spin = QDoubleSpinBox()
        self.max_capital_spin.setRange(1, 100)
        self.max_capital_spin.setValue(20)
        self.max_capital_spin.setSingleStep(5)
        self.max_capital_spin.setSuffix(" %")
        params_layout.addWidget(max_capital_label, row, col)
        params_layout.addWidget(self.max_capital_spin, row+1, col)
        
        # Action controls
        action_layout = QHBoxLayout()
        control_layout.addLayout(action_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        action_layout.addWidget(self.progress_bar, 3)
        
        # Run button
        self.run_button = QPushButton("Jalankan Backtest")
        self.run_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.run_button.clicked.connect(self.run_backtest)
        self.run_button.setEnabled(False)  # Disabled until prediction is done
        action_layout.addWidget(self.run_button, 1)
        
        # === Panel Hasil ===
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter.addWidget(results_widget)
        
        # Splitter horizontal untuk grafik dan detail
        results_splitter = QSplitter(Qt.Horizontal)
        results_layout.addWidget(results_splitter)
        
        # Panel kiri - grafik
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        results_splitter.addWidget(chart_widget)
        
        # Group box untuk grafik
        chart_group = StyledGroupBox("Hasil Backtest")
        chart_layout.addWidget(chart_group)
        
        chart_inner_layout = QVBoxLayout(chart_group)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        chart_inner_layout.addLayout(toolbar_layout)
        
        self.save_plot_button = QToolButton()
        self.save_plot_button.setText("Simpan Grafik")
        self.save_plot_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.save_plot_button.setIcon(QIcon.fromTheme("document-save"))
        self.save_plot_button.clicked.connect(self.save_backtest_plot)
        toolbar_layout.addWidget(self.save_plot_button)
        
        toolbar_layout.addStretch()
        
        # Plot area
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.figure.set_facecolor('#f9f9f9')
        self.canvas = FigureCanvas(self.figure)
        chart_inner_layout.addWidget(self.canvas)
        
        # Panel kanan - kartu hasil dan tabel
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        results_splitter.addWidget(details_widget)
        
        # Kartu hasil
        self.performance_card = ResultCard("Metrik Performa")
        details_layout.addWidget(self.performance_card)
        
        # Group box untuk tabel
        trades_group = StyledGroupBox("Riwayat Trading")
        details_layout.addWidget(trades_group)
        
        trades_layout = QVBoxLayout(trades_group)
        
        # Tabel riwayat trading
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(6)
        self.trades_table.setHorizontalHeaderLabels(["Hari", "Tanggal", "Jenis", "Harga", "Jumlah", "Nilai"])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trades_table.setAlternatingRowColors(True)
        self.trades_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f5f9ff;
                selection-background-color: #3498db;
                selection-color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
        """)
        trades_layout.addWidget(self.trades_table)
        
        # Save results button
        self.save_results_button = QPushButton("Simpan Hasil Trading")
        self.save_results_button.setIcon(QIcon.fromTheme("document-save"))
        self.save_results_button.clicked.connect(self.save_backtest_results)
        details_layout.addWidget(self.save_results_button)
        
        # Set initial splitter sizes
        main_splitter.setSizes([200, 800])
        results_splitter.setSizes([600, 400])
        
        # Disable save buttons initially
        self.save_plot_button.setEnabled(False)
        self.save_results_button.setEnabled(False)
    
    @pyqtSlot(object, object, object)
    def set_prediction_results(self, predictor, actual_prices, predicted_prices):
        """
        Set the prediction results from the prediction tab
        
        Parameters:
        -----------
        predictor : StockPredictor
            The predictor object
        actual_prices : array-like
            Actual prices from the prediction
        predicted_prices : array-like
            Predicted prices from the model
        """
        self.predictor = predictor
        self.actual_prices = actual_prices
        self.predicted_prices = predicted_prices
        self.run_button.setEnabled(True)
    
    def run_backtest(self):
        # Disable run button during backtesting
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        if self.actual_prices is None or self.predicted_prices is None:
            QMessageBox.warning(self, "Peringatan", "Silakan jalankan prediksi terlebih dahulu")
            self.run_button.setEnabled(True)
            return
        
        # Get parameters
        strategy = self.strategy_combo.currentText()
        initial_investment = self.investment_spin.value()
        commission = self.commission_spin.value() / 100  # Convert to decimal
        
        # Get risk management parameters
        max_position_size = self.position_size_spin.value() / 100  # Convert to decimal
        stop_loss = self.stop_loss_spin.value() / 100
        trailing_stop = self.trailing_stop_spin.value() / 100
        max_drawdown = self.max_drawdown_spin.value() / 100
        max_capital_per_trade = self.max_capital_spin.value() / 100
        allow_short = self.short_selling_check.isChecked()
        
        # Create and start worker thread with new parameters
        self.worker = BacktestWorker(
            self.predictor, 
            initial_investment, 
            strategy,
            transaction_fee=commission,
            allow_short=allow_short,
            max_position_size=max_position_size,
            stop_loss=stop_loss,
            trailing_stop=trailing_stop,
            max_drawdown=max_drawdown,
            max_capital_per_trade=max_capital_per_trade
        )
        self.worker.finished.connect(self.on_backtest_finished)
        self.worker.progress.connect(self.update_backtest_progress)
        self.worker.error.connect(self.show_backtest_error)
        self.worker.start()
    
    def update_backtest_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_backtest_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.run_button.setEnabled(True)
    
    def on_backtest_finished(self, portfolio_values, trades, performance):
        # Save results for later
        self.backtest_result = {
            'portfolio_values': portfolio_values,
            'trades': trades,
            'performance': performance
        }
        
        # Update plot
        self.figure.clear()
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create axes
        ax1 = self.figure.add_subplot(111)
        
        # Plot portfolio values
        days = np.arange(len(portfolio_values))
        ax1.plot(days, portfolio_values, label='Nilai Portfolio', color='#3498db', linewidth=2.5)
        
        # Highlight trades
        buy_days = [trade['day'] for trade in trades if trade['type'] == 'BUY']
        buy_values = [portfolio_values[day] for day in buy_days]
        
        sell_days = [trade['day'] for trade in trades if trade['type'] == 'SELL']
        sell_values = [portfolio_values[day] for day in sell_days]
        
        ax1.scatter(buy_days, buy_values, color='#2ecc71', s=80, marker='^', label='Beli')
        ax1.scatter(sell_days, sell_values, color='#e74c3c', s=80, marker='v', label='Jual')
        
        # Add equity curve
        if len(trades) > 0:
            # Calculate drawdowns
            rolling_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
            
            # Plot drawdown on secondary axis
            ax2 = ax1.twinx()
            ax2.fill_between(days, drawdowns, 0, alpha=0.2, color='#e74c3c', label='Drawdown')
            ax2.set_ylabel('Drawdown (%)', fontsize=10, color='#e74c3c')
            ax2.tick_params(axis='y', colors='#e74c3c')
            
            # Only show negative values on the drawdown axis
            ax2.set_ylim(min(drawdowns) * 1.5, 5)
        
        # Set labels and title
        ax1.set_title(f'Backtest Strategi {self.strategy_combo.currentText()} - {self.predictor.ticker}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hari', fontsize=10)
        ax1.set_ylabel('Nilai Portfolio (Rp)', fontsize=10, color='#3498db')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='y', colors='#3498db')
        
        # Add legend
        ax1.legend(loc='upper left', frameon=True)
        
        # Apply tight layout
        self.figure.tight_layout()
        
        # Draw canvas
        self.canvas.draw()
        
        # Update trades table
        self.trades_table.setRowCount(len(trades))
        for i, trade in enumerate(trades):
            # Create date from day index (approximate)
            trade_day_item = QTableWidgetItem(str(trade['day']))
            
            # Use predictor's start date to calculate trade date
            if hasattr(self.predictor, 'dates') and len(self.predictor.dates) > trade['day']:
                date_str = str(self.predictor.dates[trade['day']])
            else:
                date_str = f"Day {trade['day']}"
            trade_date_item = QTableWidgetItem(date_str)
            
            # Set trade type with color
            trade_type_item = QTableWidgetItem(trade['type'])
            if trade['type'] == 'BUY':
                trade_type_item.setForeground(QBrush(QColor('#2ecc71')))
                trade_type_item.setFont(QFont('Arial', 9, QFont.Bold))
            else:
                trade_type_item.setForeground(QBrush(QColor('#e74c3c')))
                trade_type_item.setFont(QFont('Arial', 9, QFont.Bold))
            
            # Format price, shares and value
            price_item = QTableWidgetItem(f"Rp {trade['price']:,.2f}")
            shares_item = QTableWidgetItem(f"{trade['shares']:.4f}")
            value_item = QTableWidgetItem(f"Rp {trade['value']:,.2f}")
            
            # Add items to table
            self.trades_table.setItem(i, 0, trade_day_item)
            self.trades_table.setItem(i, 1, trade_date_item)
            self.trades_table.setItem(i, 2, trade_type_item)
            self.trades_table.setItem(i, 3, price_item)
            self.trades_table.setItem(i, 4, shares_item)
            self.trades_table.setItem(i, 5, value_item)
        
        # Update performance metrics
        performance_text = "<table width='100%' cellspacing='5'>"
        
        # Format performance metrics in HTML
        performance_text += "<tr><td colspan='2'><b>Ringkasan Performa:</b></td></tr>"
        
        # Initial investment and ending value
        initial_investment = self.investment_spin.value()
        ending_value = portfolio_values[-1] if len(portfolio_values) > 0 else 0
        
        performance_text += f"<tr><td>Modal Awal:</td><td>Rp {initial_investment:,.2f}</td></tr>"
        performance_text += f"<tr><td>Nilai Akhir:</td><td>Rp {ending_value:,.2f}</td></tr>"
        
        # Format metrics
        for key, value in performance.items():
            if key in ['total_return', 'win_rate', 'annualized_return']:
                # Show as percentage with color
                color = '#2ecc71' if value > 0 else '#e74c3c'
                performance_text += f"<tr><td>{key.replace('_', ' ').title()}:</td><td style='color:{color}'>{value:.2f}%</td></tr>"
            elif key == 'max_drawdown':
                # Show as negative percentage in red
                performance_text += f"<tr><td>{key.replace('_', ' ').title()}:</td><td style='color:#e74c3c'>{value:.2f}%</td></tr>"
            elif key in ['profit_factor', 'sharpe_ratio']:
                # Show as float
                color = '#2ecc71' if value > 1 else '#e74c3c'
                performance_text += f"<tr><td>{key.replace('_', ' ').title()}:</td><td style='color:{color}'>{value:.2f}</td></tr>"
            elif key in ['total_profit', 'avg_profit', 'avg_loss']:
                # Show as money
                performance_text += f"<tr><td>{key.replace('_', ' ').title()}:</td><td>Rp {value:,.2f}</td></tr>"
            elif key in ['trade_count', 'win_count', 'loss_count']:
                # Show as integer
                performance_text += f"<tr><td>{key.replace('_', ' ').title()}:</td><td>{value}</td></tr>"
            else:
                # Default format
                performance_text += f"<tr><td>{key.replace('_', ' ').title()}:</td><td>{value}</td></tr>"
        
        performance_text += "</table>"
        self.performance_card.set_content(performance_text)
        
        # Enable buttons
        self.run_button.setEnabled(True)
        self.save_plot_button.setEnabled(True)
        self.save_results_button.setEnabled(True)
    
    def save_backtest_plot(self):
        """Simpan plot hasil backtest ke file"""
        if not hasattr(self, 'figure'):
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Plot Backtest", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf);;SVG (*.svg)"
        )
        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Sukses", f"Plot backtest berhasil disimpan ke {file_path}")
        
    def save_backtest_results(self):
        """Simpan hasil backtest ke file"""
        if not hasattr(self, 'backtest_result') or self.backtest_result is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Hasil Backtest", "", "CSV (*.csv);;Excel (*.xlsx)"
        )
        if file_path:
            try:
                if hasattr(self.predictor, 'save_backtest_results'):
                    self.predictor.save_backtest_results(
                        file_path, 
                        self.backtest_result['portfolio_values'],
                        self.backtest_result['trades'],
                        self.backtest_result['performance']
                    )
                    QMessageBox.information(self, "Sukses", f"Hasil backtest berhasil disimpan ke {file_path}")
                else:
                    QMessageBox.warning(self, "Peringatan", "Fungsi penyimpanan hasil tidak tersedia")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menyimpan hasil: {str(e)}") 