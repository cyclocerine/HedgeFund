"""
Stock Predictor Module
====================

Modul ini berisi implementasi kelas StockPredictor
untuk memprediksi harga saham dengan:
- Fixed data leakage (scaler fit hanya pada training data)
- Support untuk ensemble model
- Direct multi-step forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta

from ..data.preprocessor import DataPreprocessor
from .builder import ModelBuilder
from .tuner import HyperparameterTuner
from .patchtst_model import PatchTSTWrapper, ImprovedPatchTSTWrapper, patchtst_hyperparameter_search

# Try to import ensemble model
try:
    from .ensemble_model import EnsemblePredictor
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False


class TrainingProgressCallback:
    """Simple progress callback for training (pure Python, no Keras dependency)."""
    def __init__(self, total_epochs, progress_callback):
        self.total_epochs = total_epochs
        self.progress_callback = progress_callback
        
    def on_epoch_end(self, epoch, logs=None):
        progress = int((epoch + 1) / self.total_epochs * 100)
        self.progress_callback(progress, f"Melatih model (Epoch {epoch+1}/{self.total_epochs})")


class StockPredictor:
    """
    Stock Predictor dengan perbaikan:
    - Fixed data leakage: scaler fit HANYA pada training data
    - Support untuk ensemble model (PatchTST + BiLSTM + XGBoost)
    - Direct multi-step forecasting
    - Improved PatchTST dengan positional encoding
    """
    
    def __init__(self, ticker, start_date, end_date, lookback=60, forecast_days=30, 
                 model_type='patchtst', tune_hyperparameters=False, tuning_method='hyperband', 
                 log_dir=None, progress_callback=None, use_ensemble=False,
                 forecast_horizons=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.forecast_days = forecast_days
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.tuning_method = tuning_method
        self.log_dir = log_dir
        self.progress_callback = progress_callback
        self.use_ensemble = use_ensemble
        self.forecast_horizons = forecast_horizons or [20, 30, 60, 100]
        
        self.preprocessor = DataPreprocessor(ticker, start_date, end_date)
        self.scaler = MinMaxScaler()
        self.model = None
        self.patchtst_param_grid = None
        
        # Track split index untuk proper scaling
        self.train_split_idx = None
        self.scaled_data = None
        
    def prepare_data(self, train_ratio=0.8):
        """
        Mempersiapkan data untuk training dengan FIXED data leakage.
        Scaler di-fit HANYA pada training data.
        
        Args:
            train_ratio: Ratio data untuk training (default 0.8)
        """
        if self.progress_callback:
            self.progress_callback(10, "Mengunduh data...")
            
        if not self.preprocessor.download_data():
            return False
            
        if self.progress_callback:
            self.progress_callback(30, "Menghitung indikator teknikal...")
            
        if not self.preprocessor.calculate_indicators():
            return False
            
        if self.progress_callback:
            self.progress_callback(50, "Normalisasi data (training only)...")
        
        # Get features
        features = self.preprocessor.features.values if hasattr(self.preprocessor.features, 'values') else self.preprocessor.features
        
        # Calculate train split BEFORE scaling (CRITICAL: Fix data leakage)
        total_samples = len(features) - self.lookback
        self.train_split_idx = int(total_samples * train_ratio)
        
        # Fit scaler ONLY on training data portion
        train_end_idx = self.train_split_idx + self.lookback
        train_features = features[:train_end_idx]
        
        # Fit on training data only
        self.scaler.fit(train_features)
        
        # Transform ALL data using training-fitted scaler
        self.scaled_data = self.scaler.transform(features)
        
        if self.progress_callback:
            self.progress_callback(70, "Membuat sequences...")
            
        # Buat sequences
        X, y = [], []
        for i in range(self.lookback, len(self.scaled_data)):
            X.append(self.scaled_data[i-self.lookback:i])
            y.append(self.scaled_data[i, 0])
            
        self.X = np.array(X)
        self.y = np.array(y)
        
        if self.progress_callback:
            self.progress_callback(100, "Data siap!")
            
        return True
    
    def prepare_data_with_multistep(self, train_ratio=0.8, horizon=None):
        """
        Prepare data untuk direct multi-step forecasting.
        Target adalah multiple future values bukan single value.
        
        Args:
            train_ratio: Ratio for training
            horizon: Forecast horizon (e.g., 20, 30, 60)
        """
        if not self.prepare_data(train_ratio):
            return False
        
        if horizon is None:
            horizon = self.forecast_days
        
        # Create multi-step targets
        X_multi, y_multi = [], []
        
        for i in range(self.lookback, len(self.scaled_data) - horizon):
            X_multi.append(self.scaled_data[i-self.lookback:i])
            # Target: next 'horizon' values
            y_multi.append(self.scaled_data[i:i+horizon, 0])
        
        self.X = np.array(X_multi)
        self.y = np.array(y_multi)
        self.train_split_idx = int(len(self.X) * train_ratio)
        
        return True
        
    def train_model(self, epochs=50, batch_size=32, early_stopping_patience=10):
        """Melatih model dengan early stopping"""
        if self.progress_callback:
            self.progress_callback(10, "Mempersiapkan data training...")
            
        input_shape = (self.X.shape[1], self.X.shape[2])
        
        # Split data untuk validation (already prepared with proper scaling)
        X_train = self.X[:self.train_split_idx]
        X_val = self.X[self.train_split_idx:]
        y_train = self.y[:self.train_split_idx]
        y_val = self.y[self.train_split_idx:]
        
        if self.progress_callback:
            self.progress_callback(30, "Membangun model...")
        
        # Use ensemble if requested
        if self.use_ensemble and ENSEMBLE_AVAILABLE:
            self.model = EnsemblePredictor(
                input_dim=self.X.shape[2],
                forecast_horizons=self.forecast_horizons,
                progress_callback=self.progress_callback
            )
            self.model.fit(
                X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=batch_size,
                early_stopping_patience=early_stopping_patience
            )
            return None
        
        elif self.model_type in ['patchtst', 'improved_patchtst']:
            if self.tune_hyperparameters:
                param_grid = {
                    'patch_len': [8, 16, 32],
                    'stride': [4, 8, 16],
                    'd_model': [64, 128, 256],
                    'n_heads': [2, 4, 8],
                    'n_layers': [1, 2, 3],
                    'dropout': [0.1, 0.2, 0.3],
                    'lr': [0.001, 0.0005, 0.0001]
                }
                if self.patchtst_param_grid:
                    param_grid.update(self.patchtst_param_grid)
                
                # Define robust default parameters (Baseline)
                default_params = {
                    'patch_len': 16,
                    'stride': 8,
                    'd_model': 128,
                    'n_heads': 4,
                    'n_layers': 2,
                    'dropout': 0.1,
                    'lr': 0.001
                }
                
                self.model, best_params, best_score = patchtst_hyperparameter_search(
                    X_train, y_train, X_val, y_val,
                    param_grid=param_grid,
                    max_trials=10,
                    log_dir=self.log_dir,
                    progress_callback=self.progress_callback,
                    default_params=default_params
                )
            else:
                # Use improved PatchTST with positional encoding
                if self.model_type == 'improved_patchtst':
                    self.model = ImprovedPatchTSTWrapper(
                        input_dim=self.X.shape[2],
                        forecast_horizons=self.forecast_horizons,
                        progress_callback=self.progress_callback
                    )
                else:
                    self.model = PatchTSTWrapper(
                        input_dim=self.X.shape[2],
                        progress_callback=self.progress_callback
                    )
                self.model.fit(
                    X_train, y_train, X_val, y_val, 
                    epochs=epochs, batch_size=batch_size,
                    early_stopping_patience=early_stopping_patience
                )
            return None
            
        elif self.tune_hyperparameters:
            tuner = HyperparameterTuner(input_shape, tuning_method=self.tuning_method)
            self.model = tuner.tune_model(
                self.model_type,
                X_train, y_train,
                X_val, y_val
            )
        else:
            # Gunakan model standar (Keras)
            if self.model_type == 'cnn_lstm':
                self.model = ModelBuilder.build_cnn_lstm(input_shape)
            elif self.model_type == 'bilstm':
                self.model = ModelBuilder.build_bilstm(input_shape)
            elif self.model_type == 'transformer':
                self.model = ModelBuilder.build_transformer(input_shape)
            else:
                self.model = ModelBuilder.build_ensemble(input_shape)
                
            # Keras callbacks
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            if self.progress_callback:
                callbacks.append(TrainingProgressCallback(epochs, self.progress_callback))
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            if self.progress_callback:
                self.progress_callback(100, "Model selesai dilatih!")
            
            return history
        
    def predict(self, horizon=None):
        """
        Melakukan prediksi.
        
        Args:
            horizon: Forecast horizon untuk multi-step prediction
        
        Returns:
            y_original, y_pred_original, forecast_original
        """
        if self.progress_callback:
            self.progress_callback(10, "Memulai prediksi...")
        
        # Check if using ensemble or improved model
        if self.use_ensemble or isinstance(self.model, (ImprovedPatchTSTWrapper,)):
            y_pred = self.model.predict(self.X, horizon=horizon)
            if y_pred.ndim > 1:
                y_pred = y_pred[:, 0]  # Take first column for comparison
        elif self.model_type == 'patchtst':
            y_pred = self.model.predict(self.X)
        else:
            y_pred = self.model.predict(self.X)
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
        
        # Inverse transform predictions
        n_features = self.scaled_data.shape[1]
        y_pred_original = self.scaler.inverse_transform(
            np.concatenate([y_pred.reshape(-1, 1), np.zeros((len(y_pred), n_features-1))], axis=1)
        )[:, 0]
        
        if self.y.ndim > 1:
            y_flat = self.y[:, 0]
        else:
            y_flat = self.y
            
        y_original = self.scaler.inverse_transform(
            np.concatenate([y_flat.reshape(-1, 1), np.zeros((len(y_flat), n_features-1))], axis=1)
        )[:, 0]
        
        if self.progress_callback:
            self.progress_callback(50, "Membuat forecast...")
        
        # Generate forecast
        if horizon is not None and hasattr(self.model, 'predict') and isinstance(self.model, (ImprovedPatchTSTWrapper,)):
            # Direct multi-step forecasting
            last_sequence = self.X[-1:].copy()
            forecast = self.model.predict(last_sequence, horizon=horizon)
            if forecast.ndim > 1:
                forecast = forecast.flatten()
            
            forecast_original = self.scaler.inverse_transform(
                np.concatenate([forecast.reshape(-1, 1), np.zeros((len(forecast), n_features-1))], axis=1)
            )[:, 0]
        else:
            # Iterative forecasting (original method)
            last_sequence = self.X[-1].copy()
            forecast = []
            
            for i in range(self.forecast_days):
                pred = self.model.predict(last_sequence.reshape(1, *last_sequence.shape))
                if hasattr(pred, 'flatten'):
                    pred_val = pred.flatten()[0]
                else:
                    pred_val = float(pred[0])
                forecast.append(pred_val)
                
                # Update sequence
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1, 0] = pred_val
                
                if self.progress_callback:
                    progress = 50 + (i + 1) / self.forecast_days * 50
                    self.progress_callback(progress, f"Forecast hari ke-{i+1}...")
            
            forecast = np.array(forecast)
            forecast_original = self.scaler.inverse_transform(
                np.concatenate([forecast.reshape(-1, 1), np.zeros((len(forecast), n_features-1))], axis=1)
            )[:, 0]
        
        if self.progress_callback:
            self.progress_callback(100, "Prediksi selesai!")
        
        return y_original, y_pred_original, forecast_original
    
    def predict_direct_multistep(self, horizons=None):
        """
        Direct multi-step forecasting untuk multiple horizons.
        
        Args:
            horizons: List of horizons [20, 30, 60, 100]
        
        Returns:
            Dict of {horizon: forecast_original}
        """
        if horizons is None:
            horizons = self.forecast_horizons
        
        if not isinstance(self.model, (ImprovedPatchTSTWrapper, )):
            raise ValueError("Direct multi-step forecasting requires ImprovedPatchTST model")
        
        results = {}
        last_sequence = self.X[-1:].copy()
        
        n_features = self.scaled_data.shape[1]
        
        for h in horizons:
            if h in self.forecast_horizons:
                forecast = self.model.predict(last_sequence, horizon=h)
                if forecast.ndim > 1:
                    forecast = forecast.flatten()
                
                # Inverse transform
                forecast_original = self.scaler.inverse_transform(
                    np.concatenate([forecast.reshape(-1, 1), np.zeros((len(forecast), n_features-1))], axis=1)
                )[:, 0]
                
                results[h] = forecast_original
        
        return results
        
    def evaluate(self, y_true, y_pred):
        """Menghitung metrik evaluasi"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
    def plot_results(self, y_true, y_pred, forecast):
        """Plot hasil prediksi"""
        plt.figure(figsize=(15, 7))
        
        # Plot data historis
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
        
        # Plot forecast
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
        
    def save_results_to_csv(self, filename):
        """Menyimpan hasil prediksi ke CSV"""
        data = self.preprocessor.data.copy()
        
        if hasattr(self, 'y_pred'):
            data['Predicted'] = np.nan
            data.iloc[-len(self.y_pred):, data.columns.get_loc('Predicted')] = self.y_pred
            
        if hasattr(self, 'forecast'):
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(self.forecast))
            
            forecast_df = pd.DataFrame(index=forecast_dates, columns=data.columns)
            forecast_df['Predicted'] = self.forecast
            
            data = pd.concat([data, forecast_df])
            
        data.to_csv(filename)
        
    def save_results_to_excel(self, filepath):
        """Menyimpan hasil prediksi ke file Excel"""
        try:
            y_true, y_pred, forecast = self.predict()
            
            historic_df = pd.DataFrame({
                'Date': self.preprocessor.data.index[-len(y_true):],
                'Actual': y_true,
                'Predicted': y_pred,
                'Error': y_pred - y_true,
                'Error_Percent': ((y_pred - y_true) / y_true) * 100
            })
            
            forecast_dates = pd.date_range(
                start=self.preprocessor.data.index[-1] + timedelta(days=1),
                periods=len(forecast)
            )
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast
            })
            
            metrics = self.evaluate(y_true, y_pred)
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                historic_df.to_excel(writer, sheet_name='Historical Data', index=False)
                forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
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
        """Menyimpan hasil backtest ke file"""
        try:
            trades_df = pd.DataFrame(trades)
            
            portfolio_df = pd.DataFrame({
                'Day': np.arange(len(portfolio_values)),
                'Value': portfolio_values
            })
            
            performance_df = pd.DataFrame({
                'Metric': list(performance.keys()),
                'Value': list(performance.values())
            })
            
            if filepath.endswith('.csv'):
                with open(filepath, 'w') as f:
                    f.write(f"# Backtest Results\n")
                    f.write(f"# Ticker: {self.ticker}\n")
                    f.write(f"# Model: {self.model_type}\n")
                    f.write(f"# Period: {self.start_date} to {self.end_date}\n")
                    f.write(f"# Generated: {pd.Timestamp.now()}\n\n")
                    
                    f.write("## Performance Metrics\n")
                    performance_df.to_csv(f, index=False)
                    
                    f.write("\n\n## Portfolio Values\n")
                    portfolio_df.to_csv(f, index=False)
                    
                    f.write("\n\n## Trades\n")
                    trades_df.to_csv(f, index=False)
            else:
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    portfolio_df.to_excel(writer, sheet_name='Portfolio Values', index=False)
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                    performance_df.to_excel(writer, sheet_name='Performance', index=False)
                    
                    info_data = {
                        'Parameter': ['Ticker', 'Model', 'Start Date', 'End Date', 'Generated Date'],
                        'Value': [self.ticker, self.model_type, self.start_date, self.end_date, pd.Timestamp.now()]
                    }
                    pd.DataFrame(info_data).to_excel(writer, sheet_name='Info', index=False)
            
            return True
        except Exception as e:
            print(f"Error saving backtest results: {str(e)}")
            return False
            
    def get_feature_dim(self):
        """Mendapatkan dimensi fitur dari data"""
        if not hasattr(self, 'X'):
            self.prepare_data()
        return self.X.shape[2]

    def get_states_for_ppo(self):
        """Mendapatkan state untuk PPO agent"""
        states = []
        for i in range(len(self.X)):
            last_features = self.X[i, -1]
            state = np.array([
                last_features[0],  # Close price
                last_features[3],  # RSI
                last_features[4],  # MACD
                last_features[5],  # Signal Line
                last_features[8]   # BB %B
            ])
            states.append(state)
        return np.array(states)