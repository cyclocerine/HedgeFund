"""
Visualization Module
==================

Modul ini berisi fungsi-fungsi untuk visualisasi data
dan hasil prediksi.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta

def plot_stock_prediction(actual_prices, predicted_prices, forecast_prices=None, ticker=None, 
                        dates=None, forecast_dates=None, save_path=None):
    """
    Plot hasil prediksi dan forecast saham
    
    Parameters:
    -----------
    actual_prices : array-like
        Array harga aktual historis
    predicted_prices : array-like
        Array harga prediksi dari model
    forecast_prices : array-like, optional
        Array harga forecast ke depan
    ticker : str, optional
        Simbol saham untuk judul
    dates : array-like, optional
        Array tanggal untuk sumbu x
    forecast_dates : array-like, optional
        Array tanggal untuk forecast
    save_path : str, optional
        Path untuk menyimpan plot ke file
    """
    plt.figure(figsize=(15, 7))
    
    # Plot data historis
    if dates is not None:
        plt.plot(dates, actual_prices, label='Actual', color='blue')
        plt.plot(dates, predicted_prices, label='Predicted', color='red', linestyle='--')
        
        # Plot forecast jika ada
        if forecast_prices is not None and forecast_dates is not None:
            plt.plot(forecast_dates, forecast_prices, label='Forecast', color='green', linestyle='-.')
    else:
        plt.plot(actual_prices, label='Actual', color='blue')
        plt.plot(predicted_prices, label='Predicted', color='red', linestyle='--')
        
        # Plot forecast jika ada
        if forecast_prices is not None:
            plt.plot(range(len(actual_prices), len(actual_prices) + len(forecast_prices)),
                   forecast_prices, label='Forecast', color='green', linestyle='-.')
    
    # Tambahkan judul dan label
    if ticker:
        plt.title(f'{ticker} Stock Price Prediction')
    else:
        plt.title('Stock Price Prediction')
        
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Format sumbu x jika dates ada
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    # Simpan plot jika path disediakan
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Tampilkan plot
    plt.tight_layout()
    return plt.gcf()

def plot_portfolio_performance(portfolio_values, initial_investment, trades=None, 
                             dates=None, save_path=None):
    """
    Plot performa portfolio dari hasil backtesting
    
    Parameters:
    -----------
    portfolio_values : array-like
        Array nilai portfolio per hari
    initial_investment : float
        Nilai investasi awal
    trades : list, optional
        List transaksi yang dilakukan
    dates : array-like, optional
        Array tanggal untuk sumbu x
    save_path : str, optional
        Path untuk menyimpan plot ke file
    """
    plt.figure(figsize=(15, 7))
    
    # Dapatkan x axis
    if dates is not None:
        x_axis = dates
    else:
        x_axis = range(len(portfolio_values))
    
    # Plot nilai portfolio
    plt.plot(x_axis, portfolio_values, label='Portfolio Value', color='blue')
    
    # Tambahkan garis investasi awal
    plt.axhline(y=initial_investment, color='r', linestyle='--', label=f'Initial Investment (${initial_investment:,.2f})')
    
    # Plot titik transaksi jika ada
    if trades is not None and dates is not None:
        buy_dates = [dates[trade['day']] for trade in trades if trade['type'] == 'BUY']
        buy_values = [portfolio_values[trade['day']-1] for trade in trades if trade['type'] == 'BUY']
        
        sell_dates = [dates[trade['day']] for trade in trades if trade['type'] == 'SELL']
        sell_values = [portfolio_values[trade['day']-1] for trade in trades if trade['type'] == 'SELL']
        
        plt.scatter(buy_dates, buy_values, color='green', marker='^', s=100, label='Buy')
        plt.scatter(sell_dates, sell_values, color='red', marker='v', s=100, label='Sell')
    elif trades is not None:
        buy_indices = [trade['day'] for trade in trades if trade['type'] == 'BUY']
        buy_values = [portfolio_values[trade['day']-1] for trade in trades if trade['type'] == 'BUY']
        
        sell_indices = [trade['day'] for trade in trades if trade['type'] == 'SELL']
        sell_values = [portfolio_values[trade['day']-1] for trade in trades if trade['type'] == 'SELL']
        
        plt.scatter(buy_indices, buy_values, color='green', marker='^', s=100, label='Buy')
        plt.scatter(sell_indices, sell_values, color='red', marker='v', s=100, label='Sell')
    
    # Tambahkan judul dan label
    plt.title('Portfolio Performance')
    plt.xlabel('Time')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Format sumbu x jika dates ada
    if dates is not None:
        plt.gcf().autofmt_xdate()
    
    # Simpan plot jika path disediakan
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Tampilkan plot
    plt.tight_layout()
    return plt.gcf()

def plot_optimization_results(param_values, returns, param_name, strategy_name=None, save_path=None):
    """
    Plot hasil optimisasi parameter
    
    Parameters:
    -----------
    param_values : array-like
        Array nilai parameter yang diuji
    returns : array-like
        Array return untuk setiap nilai parameter
    param_name : str
        Nama parameter yang diuji
    strategy_name : str, optional
        Nama strategi untuk judul
    save_path : str, optional
        Path untuk menyimpan plot ke file
    """
    plt.figure(figsize=(12, 6))
    
    # Plot hasil
    plt.plot(param_values, returns, marker='o')
    
    # Tandai nilai parameter terbaik
    best_idx = np.argmax(returns)
    best_param = param_values[best_idx]
    best_return = returns[best_idx]
    
    plt.scatter([best_param], [best_return], color='red', s=100, zorder=5)
    plt.annotate(f'Best: {best_param:.4f}\nReturn: {best_return:.2f}%',
                xy=(best_param, best_return),
                xytext=(10, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Tambahkan judul dan label
    if strategy_name:
        plt.title(f'Parameter Optimization for {strategy_name} Strategy')
    else:
        plt.title('Parameter Optimization')
        
    plt.xlabel(param_name)
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    # Simpan plot jika path disediakan
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Tampilkan plot
    plt.tight_layout()
    return plt.gcf() 