"""
CSS Styles Module
====================

Modul ini berisi definisi style CSS untuk aplikasi.
"""

# Style sheet utama aplikasi
MAIN_STYLESHEET = """
/* Style dasar aplikasi */
QMainWindow {
    background-color: #f5f5f5;
}

QWidget {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}

/* Tab dan Panel */
QTabWidget::pane {
    border: 1px solid #ddd;
    background-color: white;
    border-radius: 5px;
}

QTabBar::tab {
    background-color: #e0e0e0;
    color: #555;
    min-width: 100px;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: white;
    color: #3498db;
    border-bottom: 2px solid #3498db;
}

QTabBar::tab:hover:!selected {
    background-color: #eaeaea;
}

/* Group Box */
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

/* Buttons */
QPushButton {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
}

QPushButton:hover {
    background-color: #2980b9;
}

QPushButton:pressed {
    background-color: #2473a6;
}

QPushButton:disabled {
    background-color: #95a5a6;
}

QToolButton {
    background-color: #ecf0f1;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 4px;
}

QToolButton:hover {
    background-color: #e0e0e0;
}

QToolButton:pressed {
    background-color: #d0d0d0;
}

/* Progress Bar */
QProgressBar {
    border: 1px solid #bdc3c7;
    border-radius: 3px;
    text-align: center;
    background-color: white;
    height: 16px;
}

QProgressBar::chunk {
    background-color: #3498db;
}

/* Input Fields */
QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QComboBox {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 4px 8px;
    background-color: white;
    selection-background-color: #3498db;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, 
QDateEdit:focus, QTimeEdit:focus, QComboBox:focus {
    border: 1px solid #3498db;
}

QComboBox::drop-down {
    border: 0px;
    width: 20px;
}

QComboBox::down-arrow {
    width: 12px;
    height: 12px;
}

/* Labels */
QLabel {
    color: #34495e;
}

/* Tables */
QTableWidget {
    background-color: white;
    alternate-background-color: #f5f9ff;
    selection-background-color: #3498db;
    selection-color: white;
    gridline-color: #ecf0f1;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
}

QHeaderView::section {
    background-color: #f0f0f0;
    padding: 6px;
    border: 1px solid #d0d0d0;
    font-weight: bold;
    color: #2c3e50;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #f0f0f0;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #bdc3c7;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #f0f0f0;
    height: 10px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #bdc3c7;
    min-width: 20px;
    border-radius: 5px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Splitters */
QSplitter::handle {
    background-color: #bdc3c7;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* Result cards */
QFrame[frameShape="6"] {
    background-color: white;
    border-radius: 5px;
    border: 1px solid #e0e0e0;
}

/* Status bar */
QStatusBar {
    background-color: #f0f0f0;
    color: #34495e;
    border-top: 1px solid #bdc3c7;
    padding: 2px;
}

/* Check boxes */
QCheckBox {
    spacing: 5px;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
}

QCheckBox::indicator:unchecked {
    border: 1px solid #bdc3c7;
    background-color: white;
    border-radius: 2px;
}

QCheckBox::indicator:checked {
    border: 1px solid #3498db;
    background-color: #3498db;
    border-radius: 2px;
}
"""

# CSS untuk tema gelap (dark mode)
DARK_STYLESHEET = """
/* Dark mode theme */
QMainWindow, QWidget {
    background-color: #2c3e50;
    color: #ecf0f1;
    font-family: 'Segoe UI', Arial, sans-serif;
}

QTabWidget::pane {
    border: 1px solid #455a64;
    background-color: #34495e;
    border-radius: 5px;
}

QTabBar::tab {
    background-color: #455a64;
    color: #bdc3c7;
    min-width: 100px;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #34495e;
    color: #3498db;
    border-bottom: 2px solid #3498db;
}

QTabBar::tab:hover:!selected {
    background-color: #546e7a;
}

QGroupBox {
    border: 1px solid #455a64;
    color: #ecf0f1;
}

QPushButton {
    background-color: #3498db;
    color: white;
}

QPushButton:hover {
    background-color: #2980b9;
}

QPushButton:disabled {
    background-color: #607d8b;
    color: #b0bec5;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QComboBox {
    background-color: #455a64;
    color: #ecf0f1;
    border: 1px solid #607d8b;
}

QTableWidget {
    background-color: #34495e;
    alternate-background-color: #2c3e50;
    color: #ecf0f1;
    gridline-color: #455a64;
    border: 1px solid #455a64;
}

QHeaderView::section {
    background-color: #455a64;
    color: #ecf0f1;
    border: 1px solid #546e7a;
}

QProgressBar {
    border: 1px solid #455a64;
    background-color: #2c3e50;
}

QFrame[frameShape="6"] {
    background-color: #34495e;
    border: 1px solid #455a64;
}

QScrollBar:vertical, QScrollBar:horizontal {
    background: #2c3e50;
}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #455a64;
}

QStatusBar {
    background-color: #2c3e50;
    color: #bdc3c7;
    border-top: 1px solid #455a64;
}
"""

# Style untuk komponen kartu hasil
CARD_STYLE = """
QFrame {
    background-color: white;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}

QLabel {
    color: #2c3e50;
}
"""

# Style untuk komponen kartu hasil dark mode
CARD_STYLE_DARK = """
QFrame {
    background-color: #34495e;
    border-radius: 8px;
    border: 1px solid #455a64;
}

QLabel {
    color: #ecf0f1;
}
""" 