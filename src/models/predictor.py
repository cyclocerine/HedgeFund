"""
Stock Predictor Module
====================

Modul ini berisi implementasi kelas StockPredictor
untuk memprediksi harga saham.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import timedelta

from ..data.preprocessor import DataPreprocessor
from .builder import ModelBuilder
from .tuner import HyperparameterTuner

class StockPredictor:
    def __init__(self, ticker, start_date, end_date, lookback=60, forecast_days=30, model_type='ensemble', tune_hyperparameters=False):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.forecast_days = forecast_days
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.preprocessor = DataPreprocessor(ticker, start_date, end_date)
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_data(self):
        """Mempersiapkan data untuk training"""
        if not self.preprocessor.download_data():
            return False
        if not self.preprocessor.calculate_indicators():
            return False
            
        # Normalisasi data
        scaled_data = self.scaler.fit_transform(self.preprocessor.features)
        
        # Buat sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])
            
        self.X = np.array(X)
        self.y = np.array(y)
        return True
        
    def train_model(self):
        """Melatih model"""
        input_shape = (self.X.shape[1], self.X.shape[2])
        
        # Split data untuk validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, shuffle=False
        )
        
        if self.tune_hyperparameters and self.model_type != 'ensemble':
            # Lakukan hyperparameter tuning
            tuner = HyperparameterTuner(input_shape)
            self.model = tuner.tune_model(
                self.model_type,
                X_train, y_train,
                X_val, y_val
            )
        else:
            # Gunakan model standar
            if self.model_type == 'cnn_lstm':
                self.model = ModelBuilder.build_cnn_lstm(input_shape)
            elif self.model_type == 'bilstm':
                self.model = ModelBuilder.build_bilstm(input_shape)
            elif self.model_type == 'transformer':
                self.model = ModelBuilder.build_transformer(input_shape)
            else:
                self.model = ModelBuilder.build_ensemble(input_shape)
                
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            # Training
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            return history
        
    def predict(self):
        """Melakukan prediksi"""
        # Prediksi pada data test
        y_pred = self.model.predict(self.X)
        
        # Transformasi kembali ke skala asli
        y_pred_original = self.scaler.inverse_transform(
            np.concatenate([y_pred, np.zeros((len(y_pred), self.X.shape[2]-1))], axis=1)
        )[:, 0]
        
        y_original = self.scaler.inverse_transform(
            np.concatenate([self.y.reshape(-1, 1), np.zeros((len(self.y), self.X.shape[2]-1))], axis=1)
        )[:, 0]
        
        # Forecasting
        last_sequence = self.X[-1]
        forecast = []
        
        for _ in range(self.forecast_days):
            pred = self.model.predict(last_sequence.reshape(1, *last_sequence.shape))
            forecast.append(pred[0, 0])
            
            # Update sequence
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = pred[0]
            
        forecast = np.array(forecast)
        forecast_original = self.scaler.inverse_transform(
            np.concatenate([forecast.reshape(-1, 1), np.zeros((len(forecast), self.X.shape[2]-1))], axis=1)
        )[:, 0]
        
        return y_original, y_pred_original, forecast_original
        
    def evaluate(self, y_true, y_pred):
        """Evaluasi model"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\nModel Evaluation:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
        
    def plot_results(self, y_true, y_pred, forecast):
        """Plot hasil prediksi"""
        plt.figure(figsize=(15, 7))
        
        # Plot data historis
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
        
        # Plot forecast
        forecast_dates = pd.date_range(
            start=self.preprocessor.data.index[-1] + timedelta(days=1),
            periods=self.forecast_days
        )
        plt.plot(
            range(len(y_true), len(y_true) + len(forecast)),
            forecast,
            label='Forecast',
            color='green',
            linestyle='-.'
        )
        
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.ticker}_prediction.png')
        plt.close()
        
    def save_results_to_csv(self, filepath):
        """
        Menyimpan hasil prediksi ke file CSV
        
        Parameters:
        -----------
        filepath : str
            Path file untuk menyimpan hasil
        """
        try:
            # Predict first to make sure we have the results
            y_true, y_pred, forecast = self.predict()
            
            # Create dataframe with historical data
            historic_df = pd.DataFrame({
                'Date': self.preprocessor.data.index[-len(y_true):],
                'Actual': y_true,
                'Predicted': y_pred,
                'Error': y_pred - y_true,
                'Error_Percent': ((y_pred - y_true) / y_true) * 100
            })
            
            # Create dataframe with forecast data
            forecast_dates = pd.date_range(
                start=self.preprocessor.data.index[-1] + timedelta(days=1),
                periods=len(forecast)
            )
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast
            })
            
            # Combine results
            results = {
                'Historic': historic_df,
                'Forecast': forecast_df,
                'Model': self.model_type,
                'Ticker': self.ticker,
                'Start_Date': self.start_date,
                'End_Date': self.end_date,
                'Lookback': self.lookback,
                'Forecast_Days': self.forecast_days
            }
            
            # Save to CSV
            with open(filepath, 'w') as f:
                # Write metadata
                f.write(f"# Stock Price Prediction Results\n")
                f.write(f"# Ticker: {self.ticker}\n")
                f.write(f"# Model: {self.model_type}\n")
                f.write(f"# Period: {self.start_date} to {self.end_date}\n")
                f.write(f"# Lookback: {self.lookback} days\n")
                f.write(f"# Forecast Days: {self.forecast_days} days\n")
                f.write(f"# Generated: {pd.Timestamp.now()}\n\n")
                
                # Write historical data
                f.write("## Historical Data\n")
                historic_df.to_csv(f, index=False)
                
                f.write("\n\n## Forecast Data\n")
                forecast_df.to_csv(f, index=False)
                
            return True
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
            return False
            
    def save_results_to_excel(self, filepath):
        """
        Menyimpan hasil prediksi ke file Excel
        
        Parameters:
        -----------
        filepath : str
            Path file untuk menyimpan hasil
        """
        try:
            # Predict first to make sure we have the results
            y_true, y_pred, forecast = self.predict()
            
            # Create dataframe with historical data
            historic_df = pd.DataFrame({
                'Date': self.preprocessor.data.index[-len(y_true):],
                'Actual': y_true,
                'Predicted': y_pred,
                'Error': y_pred - y_true,
                'Error_Percent': ((y_pred - y_true) / y_true) * 100
            })
            
            # Create dataframe with forecast data
            forecast_dates = pd.date_range(
                start=self.preprocessor.data.index[-1] + timedelta(days=1),
                periods=len(forecast)
            )
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast
            })
            
            # Create metrics data
            metrics = self.evaluate(y_true, y_pred)
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                historic_df.to_excel(writer, sheet_name='Historical Data', index=False)
                forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Create info sheet
                info_data = {
                    'Parameter': ['Ticker', 'Model', 'Start Date', 'End Date', 'Lookback Days', 
                                'Forecast Days', 'Generated Date'],
                    'Value': [self.ticker, self.model_type, self.start_date, self.end_date, 
                            self.lookback, self.forecast_days, pd.Timestamp.now()]
                }
                pd.DataFrame(info_data).to_excel(writer, sheet_name='Info', index=False)
                
            return True
        except Exception as e:
            print(f"Error saving to Excel: {str(e)}")
            return False
            
    def save_backtest_results(self, filepath, portfolio_values, trades, performance):
        """
        Menyimpan hasil backtest ke file
        
        Parameters:
        -----------
        filepath : str
            Path file untuk menyimpan hasil
        portfolio_values : array-like
            Nilai portfolio selama periode backtest
        trades : list
            Daftar transaksi trading
        performance : dict
            Metrik performa backtest
        """
        try:
            # Convert trades to dataframe
            trades_df = pd.DataFrame(trades)
            
            # Create portfolio value dataframe
            portfolio_df = pd.DataFrame({
                'Day': np.arange(len(portfolio_values)),
                'Value': portfolio_values
            })
            
            # Create performance dataframe
            performance_df = pd.DataFrame({
                'Metric': list(performance.keys()),
                'Value': list(performance.values())
            })
            
            if filepath.endswith('.csv'):
                # Save to CSV
                with open(filepath, 'w') as f:
                    # Write metadata
                    f.write(f"# Backtest Results\n")
                    f.write(f"# Ticker: {self.ticker}\n")
                    f.write(f"# Model: {self.model_type}\n")
                    f.write(f"# Period: {self.start_date} to {self.end_date}\n")
                    f.write(f"# Generated: {pd.Timestamp.now()}\n\n")
                    
                    # Write performance metrics
                    f.write("## Performance Metrics\n")
                    performance_df.to_csv(f, index=False)
                    
                    # Write portfolio values
                    f.write("\n\n## Portfolio Values\n")
                    portfolio_df.to_csv(f, index=False)
                    
                    # Write trades
                    f.write("\n\n## Trades\n")
                    trades_df.to_csv(f, index=False)
            else:
                # Save to Excel
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    portfolio_df.to_excel(writer, sheet_name='Portfolio Values', index=False)
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                    performance_df.to_excel(writer, sheet_name='Performance', index=False)
                    
                    # Create info sheet
                    info_data = {
                        'Parameter': ['Ticker', 'Model', 'Start Date', 'End Date', 'Generated Date'],
                        'Value': [self.ticker, self.model_type, self.start_date, self.end_date, pd.Timestamp.now()]
                    }
                    pd.DataFrame(info_data).to_excel(writer, sheet_name='Info', index=False)
            
            return True
        except Exception as e:
            print(f"Error saving backtest results: {str(e)}")
            return False 