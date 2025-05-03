#!/usr/bin/env python
"""
Basic Stock Prediction Example
============================

Contoh penggunaan dasar untuk prediksi harga saham.
"""

import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predictor import StockPredictor
from src.utils.visualization import plot_stock_prediction

def main():
    # Parameter dasar
    ticker = "ADRO.JK"
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    lookback = 60
    forecast_days = 30
    model_type = "ensemble"
    
    print(f"Predicting {ticker} stock prices...")
    print(f"Data range: {start_date} to {end_date}")
    print(f"Model: {model_type}")
    
    # Buat prediktor
    predictor = StockPredictor(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        lookback=lookback,
        forecast_days=forecast_days,
        model_type=model_type
    )
    
    # Persiapkan data
    print("\nPreparing data...")
    if not predictor.prepare_data():
        print("Error preparing data")
        return
    
    # Latih model
    print("\nTraining model...")
    history = predictor.train_model()
    
    # Lakukan prediksi
    print("\nMaking predictions...")
    y_true, y_pred, forecast = predictor.predict()
    
    # Evaluasi model
    print("\nEvaluating model...")
    metrics = predictor.evaluate(y_true, y_pred)
    
    # Buat hasil forecast untuk 30 hari ke depan
    print("\nForecast for the next 30 days:")
    forecast_dates = []
    start_date = predictor.preprocessor.data.index[-1]
    for i in range(1, len(forecast) + 1):
        forecast_date = start_date + timedelta(days=i)
        forecast_dates.append(forecast_date)
        print(f"Day {i} ({forecast_date.strftime('%Y-%m-%d')}): ${forecast[i-1]:.2f}")
    
    # Plot hasil
    print("\nPlotting results...")
    fig = plot_stock_prediction(
        actual_prices=y_true, 
        predicted_prices=y_pred, 
        forecast_prices=forecast,
        ticker=ticker,
        dates=predictor.preprocessor.data.index[-len(y_true):],
        forecast_dates=forecast_dates,
        save_path=f"{ticker}_prediction.png"
    )
    plt.show()

if __name__ == "__main__":
    main() 