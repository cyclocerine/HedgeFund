#!/usr/bin/env python
"""
Strategy Optimization Example
==========================

Contoh penggunaan optimisasi parameter strategi trading.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Tambahkan direktori root ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predictor import StockPredictor
from src.trading.optimizer import StrategyOptimizer
from src.trading.strategies import TradingStrategy
from src.utils.visualization import plot_optimization_results, plot_portfolio_performance
from src.utils.common import format_number, format_percentage, save_to_json

def main():
    # Parameter dasar
    ticker = "ADRO.JK"
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    lookback = 60
    model_type = "ensemble"
    initial_investment = 10000
    strategy = "Trend Following"  # Alternatif: "Mean Reversion", "Predictive"
    
    print(f"Optimizing {strategy} strategy on {ticker}...")
    print(f"Data range: {start_date} to {end_date}")
    print(f"Model: {model_type}")
    print(f"Initial investment: ${initial_investment}")
    
    # Set parameter ranges untuk optimisasi
    if strategy == "Trend Following":
        param_ranges = {
            'threshold': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        }
    elif strategy == "Mean Reversion":
        param_ranges = {
            'window': [3, 5, 7, 10, 15],
            'buy_threshold': [0.95, 0.97, 0.98, 0.99],
            'sell_threshold': [1.01, 1.02, 1.03, 1.05]
        }
    elif strategy == "Predictive":
        param_ranges = {
            'buy_threshold': [1.005, 1.01, 1.015, 1.02, 1.025],
            'sell_threshold': [0.975, 0.98, 0.985, 0.99, 0.995]
        }
    else:
        param_ranges = {}
    
    # Buat prediktor
    predictor = StockPredictor(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        lookback=lookback,
        forecast_days=0,  # Tidak perlu forecast untuk optimisasi
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
    
    # Jalankan optimisasi
    print(f"\nRunning {strategy} optimization...")
    optimizer = StrategyOptimizer(y_true, y_pred, initial_investment)
    
    # Optimalkan strategi
    best_params, best_performance, best_portfolio_values, best_trades = optimizer.optimize(
        strategy, param_ranges
    )
    
    # Tampilkan hasil
    print("\nOptimization Results:")
    print(f"Best parameters: {best_params}")
    print(f"Initial Investment: ${format_number(best_performance['initial_investment'])}")
    print(f"Final Value: ${format_number(best_performance['final_value'])}")
    print(f"Total Return: {format_number(best_performance['total_return'])}%")
    print(f"Max Drawdown: {format_number(best_performance['max_drawdown'])}%")
    print(f"Sharpe Ratio: {format_number(best_performance['sharpe_ratio'])}")
    print(f"Win Rate: {format_number(best_performance['win_rate'])}%")
    print(f"Number of Trades: {best_performance['num_trades']}")
    
    # Simpan hasil ke JSON
    results = {
        'ticker': ticker,
        'strategy': strategy,
        'best_params': best_params,
        'performance': best_performance
    }
    
    save_to_json(results, f"{ticker}_{strategy.replace(' ', '_')}_optimization_results.json")
    
    # Plot hasil optimisasi parameter
    if strategy == "Trend Following" and 'threshold' in best_params:
        param_values = param_ranges['threshold']
        returns = []
        
        # Dapatkan fungsi strategi
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        
        for threshold in param_values:
            params = {'threshold': threshold}
            _, _, performance = optimizer.run_backtest(strategy_function, params)
            returns.append(performance['total_return'])
            
        # Plot hasil optimisasi
        fig = plot_optimization_results(
            param_values=param_values,
            returns=returns,
            param_name='Threshold',
            strategy_name=strategy,
            save_path=f"{ticker}_{strategy.replace(' ', '_')}_optimization.png"
        )
        plt.figure(fig.number)
        plt.show()
    
    # Plot hasil backtesting dengan parameter optimal
    fig = plot_portfolio_performance(
        portfolio_values=best_portfolio_values,
        initial_investment=initial_investment,
        trades=best_trades,
        dates=predictor.preprocessor.data.index[-len(best_portfolio_values):],
        save_path=f"{ticker}_{strategy.replace(' ', '_')}_optimized_backtest.png"
    )
    plt.figure(fig.number)
    plt.show()

if __name__ == "__main__":
    main() 