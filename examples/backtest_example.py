#!/usr/bin/env python
"""
Backtesting Example
=================

Contoh penggunaan backtesting strategi trading.
"""

import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predictor import StockPredictor
from src.trading.backtest import Backtester
from src.utils.visualization import plot_portfolio_performance
from src.utils.common import format_number, format_percentage

def main():
    # Parameter dasar
    ticker = "ADRO.JK"
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    lookback = 60
    model_type = "ensemble"
    initial_investment = 10000
    strategy = "Trend Following"  # Alternatif: "Mean Reversion", "Predictive"
    
    print(f"Backtesting {strategy} strategy on {ticker}...")
    print(f"Data range: {start_date} to {end_date}")
    print(f"Model: {model_type}")
    print(f"Initial investment: ${initial_investment}")
    
    # Buat prediktor
    predictor = StockPredictor(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        lookback=lookback,
        forecast_days=0,  # Tidak perlu forecast untuk backtesting
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
    y_true, y_pred, _ = predictor.predict()
    
    # Evaluasi model
    print("\nEvaluating model...")
    metrics = predictor.evaluate(y_true, y_pred)
    
    # Jalankan backtesting
    print(f"\nRunning {strategy} backtest...")
    backtester = Backtester(y_true, y_pred, initial_investment)
    
    # Parameter strategi (opsional)
    if strategy == "Trend Following":
        strategy_params = {'threshold': 0.01}
    elif strategy == "Mean Reversion":
        strategy_params = {'window': 5, 'buy_threshold': 0.98, 'sell_threshold': 1.02}
    elif strategy == "Predictive":
        strategy_params = {'buy_threshold': 1.01, 'sell_threshold': 0.99}
    else:
        strategy_params = None
    
    portfolio_values, trades, performance = backtester.run(strategy, strategy_params)
    
    # Tampilkan hasil
    print("\nBacktest Results:")
    print(f"Initial Investment: ${format_number(performance['initial_investment'])}")
    print(f"Final Value: ${format_number(performance['final_value'])}")
    print(f"Total Return: {format_number(performance['total_return'])}%")
    print(f"Max Drawdown: {format_number(performance['max_drawdown'])}%")
    print(f"Sharpe Ratio: {format_number(performance['sharpe_ratio'])}")
    print(f"Win Rate: {format_number(performance['win_rate'])}%")
    print(f"Number of Trades: {performance['num_trades']}")
    
    # Tampilkan beberapa transaksi
    print("\nSample Trades:")
    for i, trade in enumerate(trades[:10]):  # Tampilkan max 10 transaksi pertama
        print(f"{i+1}. {trade['type']} on day {trade['day']}: {trade['shares']:.2f} shares at ${trade['price']:.2f}")
    
    # Plot hasil
    print("\nPlotting results...")
    fig = plot_portfolio_performance(
        portfolio_values=portfolio_values,
        initial_investment=initial_investment,
        trades=trades,
        dates=predictor.preprocessor.data.index[-len(portfolio_values):],
        save_path=f"{ticker}_{strategy.replace(' ', '_')}_backtest.png"
    )
    plt.show()

if __name__ == "__main__":
    main() 