"""
Portfolio Tab Module
===================

Modul ini berisi implementasi tab portofolio multi-aset untuk UI aplikasi.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QDoubleSpinBox, QMessageBox, QTableWidget, 
                             QTableWidgetItem, QFrame, QSplitter, QGroupBox, QHeaderView, 
                             QLineEdit, QGridLayout, QTabWidget, QCheckBox, QToolButton)
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.QtGui import QColor, QBrush, QFont, QIcon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

from .prediction_tab import StyledGroupBox, ResultCard

class PortfolioTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.portfolio = None
        self.current_prices = {}
        self.setup_ui()
        
    def setup_ui(self):
        # Layout utama
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Splitter utama untuk membagi panel kontrol dan hasil
        main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(main_splitter)
        
        # === Panel Kontrol ===
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter.addWidget(control_widget)
        
        # Sub-tabs untuk Asset Management dan Risk Settings
        tabs = QTabWidget()
        control_layout.addWidget(tabs)
        
        # Tab Asset Management
        asset_tab = QWidget()
        asset_layout = QVBoxLayout(asset_tab)
        tabs.addTab(asset_tab, "Manajemen Aset")
        
        # Panel untuk menambahkan aset
        add_asset_group = StyledGroupBox("Tambah Aset")
        asset_layout.addWidget(add_asset_group)
        
        add_asset_layout = QGridLayout(add_asset_group)
        
        # Ticker input
        ticker_label = QLabel("Kode Saham:")
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Contoh: ADRO.JK, MSFT")
        add_asset_layout.addWidget(ticker_label, 0, 0)
        add_asset_layout.addWidget(self.ticker_input, 0, 1)
        
        # Asset type
        asset_type_label = QLabel("Jenis Aset:")
        self.asset_type_combo = QComboBox()
        self.asset_type_combo.addItems(["Saham", "Forex", "Crypto", "Komoditas", "ETF"])
        add_asset_layout.addWidget(asset_type_label, 0, 2)
        add_asset_layout.addWidget(self.asset_type_combo, 0, 3)
        
        # Add button
        self.add_asset_button = QPushButton("Tambah Aset")
        self.add_asset_button.clicked.connect(self.add_asset)
        add_asset_layout.addWidget(self.add_asset_button, 0, 4)
        
        # Panel untuk mengatur modal awal
        capital_group = StyledGroupBox("Modal dan Alokasi")
        asset_layout.addWidget(capital_group)
        
        capital_layout = QGridLayout(capital_group)
        
        # Initial capital
        capital_label = QLabel("Modal Awal:")
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000000, 1000000000)
        self.capital_spin.setValue(100000000)
        self.capital_spin.setSingleStep(1000000)
        self.capital_spin.setPrefix("Rp ")
        self.capital_spin.setGroupSeparatorShown(True)
        capital_layout.addWidget(capital_label, 0, 0)
        capital_layout.addWidget(self.capital_spin, 0, 1)
        
        # Transaction fee
        fee_label = QLabel("Biaya Transaksi (%):")
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0, 2)
        self.fee_spin.setValue(0.15)
        self.fee_spin.setSingleStep(0.05)
        self.fee_spin.setSuffix(" %")
        capital_layout.addWidget(fee_label, 0, 2)
        capital_layout.addWidget(self.fee_spin, 0, 3)
        
        # Initialize button
        self.init_portfolio_button = QPushButton("Inisialisasi Portofolio")
        self.init_portfolio_button.clicked.connect(self.initialize_portfolio)
        capital_layout.addWidget(self.init_portfolio_button, 0, 4)
        
        # Tab Risk Management
        risk_tab = QWidget()
        risk_layout = QVBoxLayout(risk_tab)
        tabs.addTab(risk_tab, "Manajemen Risiko")
        
        # Risk parameters group
        risk_params_group = StyledGroupBox("Parameter Risiko")
        risk_layout.addWidget(risk_params_group)
        
        risk_params_layout = QGridLayout(risk_params_group)
        
        # Max position size
        pos_size_label = QLabel("Ukuran Posisi Maks (%):")
        self.pos_size_spin = QDoubleSpinBox()
        self.pos_size_spin.setRange(1, 100)
        self.pos_size_spin.setValue(20)
        self.pos_size_spin.setSingleStep(5)
        self.pos_size_spin.setSuffix(" %")
        risk_params_layout.addWidget(pos_size_label, 0, 0)
        risk_params_layout.addWidget(self.pos_size_spin, 0, 1)
        
        # Stop loss
        stop_loss_label = QLabel("Stop Loss (%):")
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.5, 20)
        self.stop_loss_spin.setValue(5)
        self.stop_loss_spin.setSingleStep(0.5)
        self.stop_loss_spin.setSuffix(" %")
        risk_params_layout.addWidget(stop_loss_label, 0, 2)
        risk_params_layout.addWidget(self.stop_loss_spin, 0, 3)
        
        # Trailing stop
        trailing_stop_label = QLabel("Trailing Stop (%):")
        self.trailing_stop_spin = QDoubleSpinBox()
        self.trailing_stop_spin.setRange(0.5, 20)
        self.trailing_stop_spin.setValue(3)
        self.trailing_stop_spin.setSingleStep(0.5)
        self.trailing_stop_spin.setSuffix(" %")
        risk_params_layout.addWidget(trailing_stop_label, 1, 0)
        risk_params_layout.addWidget(self.trailing_stop_spin, 1, 1)
        
        # Max drawdown
        max_dd_label = QLabel("Max Drawdown (%):")
        self.max_dd_spin = QDoubleSpinBox()
        self.max_dd_spin.setRange(5, 50)
        self.max_dd_spin.setValue(10)
        self.max_dd_spin.setSingleStep(1)
        self.max_dd_spin.setSuffix(" %")
        risk_params_layout.addWidget(max_dd_label, 1, 2)
        risk_params_layout.addWidget(self.max_dd_spin, 1, 3)
        
        # Max capital per trade
        max_capital_label = QLabel("Kapital per Trade (%):")
        self.max_capital_spin = QDoubleSpinBox()
        self.max_capital_spin.setRange(1, 100)
        self.max_capital_spin.setValue(20)
        self.max_capital_spin.setSingleStep(5)
        self.max_capital_spin.setSuffix(" %")
        risk_params_layout.addWidget(max_capital_label, 2, 0)
        risk_params_layout.addWidget(self.max_capital_spin, 2, 1)
        
        # Short selling allowed
        short_label = QLabel("Short Selling:")
        self.short_check = QCheckBox("Aktifkan")
        risk_params_layout.addWidget(short_label, 2, 2)
        risk_params_layout.addWidget(self.short_check, 2, 3)
        
        # Apply risk settings button
        self.apply_risk_button = QPushButton("Terapkan Pengaturan")
        self.apply_risk_button.clicked.connect(self.apply_risk_settings)
        risk_params_layout.addWidget(self.apply_risk_button, 3, 0, 1, 4)
        
        # Trading panel group
        trading_group = StyledGroupBox("Eksekusi Trading")
        control_layout.addWidget(trading_group)
        
        trading_layout = QGridLayout(trading_group)
        
        # Symbol selection
        symbol_label = QLabel("Saham:")
        self.symbol_combo = QComboBox()
        trading_layout.addWidget(symbol_label, 0, 0)
        trading_layout.addWidget(self.symbol_combo, 0, 1)
        
        # Action selection
        action_label = QLabel("Aksi:")
        self.action_combo = QComboBox()
        self.action_combo.addItems(["BUY", "SELL", "SHORT", "COVER"])
        trading_layout.addWidget(action_label, 0, 2)
        trading_layout.addWidget(self.action_combo, 0, 3)
        
        # Size selection
        size_label = QLabel("Ukuran (%):")
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_spin.setValue(10)
        self.size_spin.setSingleStep(5)
        self.size_spin.setSuffix(" %")
        trading_layout.addWidget(size_label, 0, 4)
        trading_layout.addWidget(self.size_spin, 0, 5)
        
        # Execute button
        self.execute_button = QPushButton("Eksekusi")
        self.execute_button.clicked.connect(self.execute_trade)
        trading_layout.addWidget(self.execute_button, 0, 6)
        
        # Update prices button
        self.update_prices_button = QPushButton("Update Harga")
        self.update_prices_button.clicked.connect(self.update_prices)
        trading_layout.addWidget(self.update_prices_button, 0, 7)
        
        # === Panel Hasil ===
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter.addWidget(results_widget)
        
        # Splitter horizontal untuk tabel dan grafik
        results_splitter = QSplitter(Qt.Horizontal)
        results_layout.addWidget(results_splitter)
        
        # Left panel - dashboard
        dashboard_widget = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_widget)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)
        results_splitter.addWidget(dashboard_widget)
        
        # Portfolio summary
        self.portfolio_card = ResultCard("Ringkasan Portofolio")
        dashboard_layout.addWidget(self.portfolio_card)
        
        # Active positions group
        positions_group = StyledGroupBox("Posisi Aktif")
        dashboard_layout.addWidget(positions_group)
        
        positions_layout = QVBoxLayout(positions_group)
        
        # Positions table
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(6)
        self.positions_table.setHorizontalHeaderLabels(["Saham", "Tipe", "Harga Masuk", "Harga Saat Ini", "Jumlah", "P/L (%)"])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.positions_table.setAlternatingRowColors(True)
        positions_layout.addWidget(self.positions_table)
        
        # Right panel - charts and transactions
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        results_splitter.addWidget(charts_widget)
        
        # Portfolio performance chart
        chart_group = StyledGroupBox("Kinerja Portofolio")
        charts_layout.addWidget(chart_group)
        
        chart_layout = QVBoxLayout(chart_group)
        
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.set_facecolor('#f9f9f9')
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        
        # Transactions group
        transactions_group = StyledGroupBox("Riwayat Transaksi")
        charts_layout.addWidget(transactions_group)
        
        transactions_layout = QVBoxLayout(transactions_group)
        
        # Transactions table
        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(6)
        self.transactions_table.setHorizontalHeaderLabels(["Tanggal", "Saham", "Aksi", "Harga", "Jumlah", "Nilai"])
        self.transactions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.transactions_table.setAlternatingRowColors(True)
        transactions_layout.addWidget(self.transactions_table)
        
        # Set initial splitter sizes
        main_splitter.setSizes([300, 700])
        results_splitter.setSizes([500, 500])
        
        # Disable buttons until portfolio is initialized
        self.execute_button.setEnabled(False)
        self.update_prices_button.setEnabled(False)
        self.apply_risk_button.setEnabled(False)
        
    def add_asset(self):
        """Menambahkan aset ke daftar"""
        ticker = self.ticker_input.text().strip()
        if not ticker:
            QMessageBox.warning(self, "Peringatan", "Silakan masukkan kode saham")
            return
            
        # Cek apakah ticker sudah ada di combo box
        if self.symbol_combo.findText(ticker) == -1:
            self.symbol_combo.addItem(ticker)
            self.ticker_input.clear()
            
    def initialize_portfolio(self):
        """Inisialisasi portofolio"""
        from src.trading.portfolio import MultiAssetPortfolio
        
        # Dapatkan daftar aset
        assets = [self.symbol_combo.itemText(i) for i in range(self.symbol_combo.count())]
        
        if not assets:
            QMessageBox.warning(self, "Peringatan", "Silakan tambahkan minimal satu aset")
            return
            
        # Dapatkan modal awal dan fee
        initial_capital = self.capital_spin.value()
        transaction_fee = self.fee_spin.value() / 100  # Convert to decimal
        
        # Inisialisasi portfolio
        self.portfolio = MultiAssetPortfolio(
            assets=assets,
            initial_capital=initial_capital,
            transaction_fee=transaction_fee
        )
        
        # Apply risk settings
        self.apply_risk_settings()
        
        # Enable buttons
        self.execute_button.setEnabled(True)
        self.update_prices_button.setEnabled(True)
        self.apply_risk_button.setEnabled(True)
        
        # Update UI
        self.update_portfolio_display()
        
        QMessageBox.information(self, "Success", f"Portofolio berhasil diinisialisasi dengan modal Rp {initial_capital:,.0f}")
        
    def apply_risk_settings(self):
        """Terapkan pengaturan risk management"""
        if not self.portfolio:
            QMessageBox.warning(self, "Peringatan", "Silakan inisialisasi portofolio terlebih dahulu")
            return
            
        # Get risk parameters
        max_position_size = self.pos_size_spin.value() / 100
        stop_loss = self.stop_loss_spin.value() / 100
        trailing_stop = self.trailing_stop_spin.value() / 100
        max_drawdown = self.max_dd_spin.value() / 100
        max_capital_per_trade = self.max_capital_spin.value() / 100
        
        # Update risk manager
        from src.trading.risk_manager import RiskManager
        self.portfolio.risk_manager = RiskManager(
            max_drawdown=max_drawdown,
            max_position_size=max_position_size,
            stop_loss=stop_loss,
            trailing_stop=trailing_stop,
            max_capital_per_trade=max_capital_per_trade
        )
        
        QMessageBox.information(self, "Success", "Parameter manajemen risiko berhasil diterapkan")
        
    def update_prices(self):
        """Update harga aset dari Yahoo Finance"""
        if not self.portfolio:
            QMessageBox.warning(self, "Peringatan", "Silakan inisialisasi portofolio terlebih dahulu")
            return
            
        try:
            import yfinance as yf
            
            # Dapatkan daftar aset
            assets = self.portfolio.assets
            
            # Get current prices
            self.current_prices = {}
            for symbol in assets:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d")
                    
                    if not data.empty:
                        # Get the latest closing price
                        self.current_prices[symbol] = data['Close'].iloc[-1]
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Gagal mendapatkan data untuk {symbol}: {str(e)}")
            
            # Update portfolio with new prices
            if self.current_prices:
                self.portfolio.update_prices(self.current_prices, timestamp=datetime.now())
                self.update_portfolio_display()
                QMessageBox.information(self, "Success", f"Harga berhasil diupdate untuk {len(self.current_prices)} aset")
            else:
                QMessageBox.warning(self, "Peringatan", "Tidak ada harga yang berhasil diupdate")
                
        except ImportError:
            QMessageBox.critical(self, "Error", "yfinance library not found. Please install it with: pip install yfinance")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating prices: {str(e)}")
            
    def execute_trade(self):
        """Eksekusi perdagangan"""
        if not self.portfolio:
            QMessageBox.warning(self, "Peringatan", "Silakan inisialisasi portofolio terlebih dahulu")
            return
            
        if not self.current_prices:
            QMessageBox.warning(self, "Peringatan", "Silakan update harga terlebih dahulu")
            return
            
        # Get parameters
        symbol = self.symbol_combo.currentText()
        action = self.action_combo.currentText()
        size = self.size_spin.value() / 100  # Convert to decimal
        
        # Check if symbol has price
        if symbol not in self.current_prices:
            QMessageBox.warning(self, "Peringatan", f"Tidak ada harga untuk {symbol}. Silakan update harga.")
            return
            
        # Check if action is valid for current positions
        if action == "SELL" and (symbol not in self.portfolio.positions or 
                                 self.portfolio.positions[symbol]['type'] != 'LONG'):
            QMessageBox.warning(self, "Peringatan", f"Tidak ada posisi LONG untuk {symbol} yang dapat dijual")
            return
            
        if action == "COVER" and (symbol not in self.portfolio.positions or 
                                  self.portfolio.positions[symbol]['type'] != 'SHORT'):
            QMessageBox.warning(self, "Peringatan", f"Tidak ada posisi SHORT untuk {symbol} yang dapat di-cover")
            return
            
        if (action == "BUY" or action == "SHORT") and symbol in self.portfolio.positions:
            QMessageBox.warning(self, "Peringatan", f"Sudah ada posisi untuk {symbol}. Tutup posisi terlebih dahulu.")
            return
            
        # Create signal
        signal = [{
            'symbol': symbol,
            'action': action,
            'size': size,
            'volatility': 0.02  # Default volatility
        }]
        
        # Execute trade
        result = self.portfolio.allocate_capital(signal, self.current_prices, timestamp=datetime.now())
        
        # Update display
        self.update_portfolio_display()
        
        # Show result
        if result['executed_orders']:
            QMessageBox.information(self, "Success", f"Order berhasil dieksekusi: {action} {symbol}")
        else:
            QMessageBox.warning(self, "Peringatan", "Tidak ada order yang dieksekusi")
            
    def update_portfolio_display(self):
        """Update tampilan portofolio"""
        if not self.portfolio:
            return
            
        # Update portfolio summary card
        summary = self.portfolio.get_portfolio_summary()
        
        # Format values for display
        self.portfolio_card.clear_results()
        self.portfolio_card.add_result("Nilai Portofolio", f"Rp {summary['current_value']:,.0f}")
        self.portfolio_card.add_result("Uang Tunai", f"Rp {summary['cash']:,.0f}")
        self.portfolio_card.add_result("Nilai Investasi", f"Rp {summary['invested_value']:,.0f}")
        self.portfolio_card.add_result("Return", f"{summary['return_pct']:.2f}%")
        
        # Update positions table
        positions_df = self.portfolio.get_positions_df()
        self.positions_table.setRowCount(0)
        
        if not positions_df.empty:
            self.positions_table.setRowCount(len(positions_df))
            
            for i, (_, row) in enumerate(positions_df.iterrows()):
                self.positions_table.setItem(i, 0, QTableWidgetItem(row['symbol']))
                self.positions_table.setItem(i, 1, QTableWidgetItem(row['type']))
                self.positions_table.setItem(i, 2, QTableWidgetItem(f"{row['entry_price']:.2f}"))
                
                current_price = self.current_prices.get(row['symbol'], 0)
                self.positions_table.setItem(i, 3, QTableWidgetItem(f"{current_price:.2f}"))
                self.positions_table.setItem(i, 4, QTableWidgetItem(f"{row['shares']:.4f}"))
                
                pnl_pct = row.get('unrealized_pnl_pct', 0)
                pnl_item = QTableWidgetItem(f"{pnl_pct:.2f}%")
                
                # Set color based on P/L
                if pnl_pct > 0:
                    pnl_item.setForeground(QBrush(QColor("green")))
                elif pnl_pct < 0:
                    pnl_item.setForeground(QBrush(QColor("red")))
                    
                self.positions_table.setItem(i, 5, pnl_item)
                
        # Update transactions table
        transactions_df = self.portfolio.get_transactions_df()
        self.transactions_table.setRowCount(0)
        
        if not transactions_df.empty:
            self.transactions_table.setRowCount(len(transactions_df))
            
            for i, (_, row) in enumerate(transactions_df.iterrows()):
                # Format timestamp
                if isinstance(row['timestamp'], datetime):
                    timestamp = row['timestamp'].strftime("%Y-%m-%d %H:%M")
                else:
                    timestamp = str(row['timestamp'])
                    
                self.transactions_table.setItem(i, 0, QTableWidgetItem(timestamp))
                self.transactions_table.setItem(i, 1, QTableWidgetItem(row['symbol']))
                
                action_item = QTableWidgetItem(row['action'])
                
                # Set color based on action
                if row['action'] in ['BUY', 'COVER']:
                    action_item.setForeground(QBrush(QColor("green")))
                elif row['action'] in ['SELL', 'SHORT']:
                    action_item.setForeground(QBrush(QColor("red")))
                    
                self.transactions_table.setItem(i, 2, action_item)
                self.transactions_table.setItem(i, 3, QTableWidgetItem(f"{row['price']:.2f}"))
                self.transactions_table.setItem(i, 4, QTableWidgetItem(f"{row['shares']:.4f}"))
                self.transactions_table.setItem(i, 5, QTableWidgetItem(f"Rp {row['value']:,.0f}"))
                
        # Update portfolio performance chart
        history_df = self.portfolio.get_portfolio_history_df()
        
        if not history_df.empty:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot portfolio value over time
            ax.plot(history_df['timestamp'], history_df['value'], 'b-', label='Nilai Portofolio')
            
            # Format x-axis to show dates nicely
            if len(history_df) > 1:
                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Nilai (Rp)')
                ax.set_title('Kinerja Portofolio')
                ax.grid(True)
                ax.legend()
                
                # Format y-axis with commas for thousands
                import matplotlib.ticker as mtick
                ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
                
                # Rotate date labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                self.figure.tight_layout()
                
            self.canvas.draw() 