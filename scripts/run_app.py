#!/usr/bin/env python
"""
GUI Application Runner
====================

Script untuk menjalankan aplikasi prediksi saham dengan antarmuka grafis.
"""

import sys
import os
import argparse

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QLocale
from src.gui.app import StockPredictionApp
from src.gui.styles import MAIN_STYLESHEET, DARK_STYLESHEET

def main():
    """
    Fungsi utama untuk menjalankan aplikasi GUI
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Price Prediction GUI')
    parser.add_argument('--dark', action='store_true', help='Gunakan tema gelap')
    parser.add_argument('--style', type=str, choices=['fusion', 'windows', 'windowsvista'], 
                        default='fusion', help='Pilih style aplikasi')
    args = parser.parse_args()
    
    # Inisialisasi aplikasi
    app = QApplication(sys.argv)
    
    # Set locale ke Indonesia
    locale = QLocale(QLocale.Indonesian, QLocale.Indonesia)
    QLocale.setDefault(locale)
    
    # Set style sheet
    if args.dark:
        app.setStyleSheet(DARK_STYLESHEET)
    else:
        app.setStyleSheet(MAIN_STYLESHEET)
    
    # Buat jendela aplikasi
    window = StockPredictionApp(use_dark_theme=args.dark, style_name=args.style)
    window.show()
    
    # Jalankan event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 