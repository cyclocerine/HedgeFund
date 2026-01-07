#!/usr/bin/env python
"""
Model Evaluation Script
=======================

Script untuk mengevaluasi kemampuan prediksi model PatchTST dengan:
- Menggunakan data 10 tahun
- Melatih model hanya dengan 5 tahun pertama
- Memprediksi 20, 30, dan 60 hari ke depan
- Membandingkan dengan data aktual untuk menilai apakah model benar-benar memprediksi
  atau hanya menghafal pola data
"""

import argparse
from datetime import datetime, timedelta
import sys
import os

# Must be set before any other imports to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("[!] Rich library tidak tersedia. Menggunakan output standar.")

from src.models.predictor import StockPredictor
from src.models.patchtst_model import PatchTSTWrapper
from src.data.preprocessor import DataPreprocessor
from sklearn.preprocessing import MinMaxScaler

console = Console() if RICH_AVAILABLE else None


def print_header():
    if RICH_AVAILABLE:
        header = """
╔════════════════════════════════════════════════════════════════╗
║  MODEL EVALUATION - Prediksi vs Menghafal                      ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║
║  10 Tahun Data | 5 Tahun Training | Multi-Horizon Testing      ║
╚════════════════════════════════════════════════════════════════╝
"""
        console.print(header, style="bright_cyan")
    else:
        print("=" * 60)
        print("  MODEL EVALUATION - Prediksi vs Menghafal")
        print("  10 Tahun Data | 5 Tahun Training | Multi-Horizon Testing")
        print("=" * 60)


def print_info(message):
    if RICH_AVAILABLE:
        console.print(f"[INFO] {message}", style="bright_blue")
    else:
        print(f"[INFO] {message}")


def print_success(message):
    if RICH_AVAILABLE:
        console.print(f"[OK] {message}", style="bright_green")
    else:
        print(f"[OK] {message}")


def print_warning(message):
    if RICH_AVAILABLE:
        console.print(f"[WARN] {message}", style="bright_yellow")
    else:
        print(f"[WARN] {message}")


def print_error(message):
    if RICH_AVAILABLE:
        console.print(f"[ERROR] {message}", style="bold red")
    else:
        print(f"[ERROR] {message}")


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy - persentase model benar memprediksi arah harga.
    Ini penting untuk mendeteksi jika model hanya 'mengikuti' dengan delay.
    """
    if len(y_true) < 2:
        return 0.0
    
    actual_direction = np.diff(y_true) > 0  # True = naik, False = turun
    pred_direction = np.diff(y_pred) > 0
    
    # Match directions
    correct = np.sum(actual_direction == pred_direction)
    total = len(actual_direction)
    
    return (correct / total) * 100


def calculate_lag_correlation(y_true, y_pred, max_lag=5):
    """
    Calculate correlation at different lags to detect if model is just lagging behind.
    If correlation is highest at lag > 0, model might just be copying past values.
    """
    correlations = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            corr = np.corrcoef(y_true[lag:], y_pred[:-lag])[0, 1]
        correlations.append((lag, corr))
    return correlations


class ModelEvaluator:
    """Evaluator untuk menguji kemampuan prediksi model"""
    
    def __init__(self, ticker, lookback=60, verbose=True):
        self.ticker = ticker
        self.lookback = lookback
        self.verbose = verbose
        self.scaler = MinMaxScaler()
        self.model = None
        self.results = {}
        
    def load_and_split_data(self, total_years=10, train_years=5):
        """Load data dan split menjadi training dan test set"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_years * 365)
        split_date = start_date + timedelta(days=train_years * 365)
        
        if self.verbose:
            print_info(f"Mengunduh data {self.ticker} dari {start_date.strftime('%Y-%m-%d')} hingga {end_date.strftime('%Y-%m-%d')}")
        
        # Download and prepare data
        self.preprocessor = DataPreprocessor(
            self.ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not self.preprocessor.download_data():
            print_error("Gagal mengunduh data")
            return False
            
        if not self.preprocessor.calculate_indicators():
            print_error("Gagal menghitung indikator")
            return False
        
        # Get the data with dates
        data = self.preprocessor.data
        features = self.preprocessor.features
        
        if self.verbose:
            print_success(f"Data berhasil diunduh: {len(data)} baris")
        
        # Split by date
        self.split_idx = len(data[data.index <= split_date])
        
        if self.verbose:
            print_info(f"Training data: {self.split_idx} hari (hingga {split_date.strftime('%Y-%m-%d')})")
            print_info(f"Test data: {len(data) - self.split_idx} hari")
        
        # Scale data - ensure features is a 2D numpy array
        features_array = features.values if hasattr(features, 'values') else np.array(features)
        self.scaled_data = self.scaler.fit_transform(features_array)
        
        # Store raw prices for comparison - handle both single and MultiIndex columns
        if 'Close' in data.columns:
            close_data = data['Close']
        elif ('Close', self.ticker) in data.columns:
            close_data = data[('Close', self.ticker)]
        else:
            # Try to find Close in features
            if 'Close' in features.columns:
                close_data = features['Close']
            else:
                # Fallback: use first column of features (should be Close)
                close_data = features.iloc[:, 0]
        
        self.all_prices = close_data.values.flatten() if hasattr(close_data, 'values') else np.array(close_data).flatten()
        self.all_dates = data.index
        
        return True
    
    def prepare_sequences(self, data, lookback):
        """Prepare sequences for training/prediction"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])  # Close price is first column
        return np.array(X), np.array(y)
    
    def train_model(self, epochs=50, batch_size=32):
        """Train model hanya dengan data training"""
        if self.verbose:
            print_info("Mempersiapkan data training...")
        
        # Prepare training sequences (only from training data)
        train_data = self.scaled_data[:self.split_idx]
        X_train, y_train = self.prepare_sequences(train_data, self.lookback)
        
        # Split for validation
        val_split = int(len(X_train) * 0.8)
        X_t, X_v = X_train[:val_split], X_train[val_split:]
        y_t, y_v = y_train[:val_split], y_train[val_split:]
        
        if self.verbose:
            print_info(f"Training samples: {len(X_t)}, Validation samples: {len(X_v)}")
            print_info("Melatih model PatchTST...")
        
        start_time = time.time()
        
        # Train PatchTST
        self.model = PatchTSTWrapper(
            input_dim=X_train.shape[2],
            patch_len=16,
            stride=8,
            d_model=128,
            n_heads=4,
            n_layers=2,
            dropout=0.2
        )
        
        self.model.fit(X_t, y_t, X_v, y_v, epochs=epochs, batch_size=batch_size)
        
        training_time = time.time() - start_time
        
        if self.verbose:
            print_success(f"Model selesai dilatih dalam {training_time:.1f} detik")
        
        # Store training sequences for later use
        self.X_train = X_train
        self.y_train = y_train
        
        return True
    
    def evaluate_forecast_horizon(self, forecast_days, num_windows=10):
        """
        Evaluate model performance for a specific forecast horizon.
        Uses multiple prediction windows across the test period.
        """
        if self.verbose:
            print_info(f"Mengevaluasi horizon {forecast_days} hari...")
        
        # Get test data
        test_start = self.split_idx
        test_data = self.scaled_data[test_start - self.lookback:]
        test_prices = self.all_prices[test_start:]
        
        # Calculate window positions
        available_days = len(test_prices) - forecast_days
        if available_days < num_windows:
            num_windows = max(1, available_days)
        
        window_step = max(1, available_days // num_windows)
        
        all_predictions = []
        all_actuals = []
        all_errors = []
        
        for w in range(num_windows):
            window_start = w * window_step
            
            if window_start + forecast_days > len(test_data) - self.lookback:
                break
            
            # Get the sequence at this window
            seq_start = window_start
            sequence = test_data[seq_start:seq_start + self.lookback].copy()
            
            # Make predictions for forecast_days ahead
            predictions = []
            current_seq = sequence.copy()
            
            for day in range(forecast_days):
                pred = self.model.predict(current_seq.reshape(1, *current_seq.shape))
                predictions.append(pred[0])
                
                # Roll sequence and add prediction
                current_seq = np.roll(current_seq, -1, axis=0)
                current_seq[-1] = pred
            
            predictions = np.array(predictions)
            
            # Inverse transform predictions
            pred_prices = self.scaler.inverse_transform(
                np.concatenate([predictions.reshape(-1, 1), 
                               np.zeros((len(predictions), self.scaled_data.shape[1]-1))], axis=1)
            )[:, 0]
            
            # Get actual prices for this window
            actual_start = window_start
            actual_end = actual_start + forecast_days
            actual_prices = test_prices[actual_start:actual_end]
            
            if len(actual_prices) == len(pred_prices):
                all_predictions.extend(pred_prices)
                all_actuals.extend(actual_prices)
                all_errors.extend(pred_prices - actual_prices)
        
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        if len(all_predictions) == 0:
            return None
        
        # Calculate metrics
        metrics = {
            'forecast_days': forecast_days,
            'num_samples': len(all_predictions),
            'mse': mean_squared_error(all_actuals, all_predictions),
            'rmse': np.sqrt(mean_squared_error(all_actuals, all_predictions)),
            'mae': mean_absolute_error(all_actuals, all_predictions),
            'mape': calculate_mape(all_actuals, all_predictions),
            'r2': r2_score(all_actuals, all_predictions),
            'directional_accuracy': calculate_directional_accuracy(all_actuals, all_predictions),
            'predictions': all_predictions,
            'actuals': all_actuals,
            'errors': all_errors
        }
        
        # Calculate lag correlations
        lag_corrs = calculate_lag_correlation(all_actuals, all_predictions, max_lag=5)
        metrics['lag_correlations'] = lag_corrs
        
        # Find best lag
        best_lag = max(lag_corrs, key=lambda x: x[1] if not np.isnan(x[1]) else -1)
        metrics['best_lag'] = best_lag[0]
        metrics['best_lag_corr'] = best_lag[1]
        
        return metrics
    
    def run_full_evaluation(self, horizons=[20, 30, 60]):
        """Run evaluation for multiple forecast horizons"""
        self.results = {}
        
        for horizon in horizons:
            metrics = self.evaluate_forecast_horizon(horizon)
            if metrics:
                self.results[horizon] = metrics
        
        return self.results
    
    def print_results(self):
        """Print evaluation results in a nice table"""
        if not self.results:
            print_warning("Tidak ada hasil untuk ditampilkan")
            return
        
        if RICH_AVAILABLE:
            # Main metrics table
            table = Table(title="📊 Hasil Evaluasi Model", box=box.ROUNDED, border_style="bright_blue")
            table.add_column("Metrik", style="cyan")
            
            for horizon in sorted(self.results.keys()):
                table.add_column(f"{horizon} Hari", style="green", justify="right")
            
            metrics_to_show = [
                ('Samples', 'num_samples', '{:.0f}'),
                ('MSE', 'mse', '{:.4f}'),
                ('RMSE', 'rmse', '{:.4f}'),
                ('MAE', 'mae', '{:.4f}'),
                ('MAPE (%)', 'mape', '{:.2f}%'),
                ('R² Score', 'r2', '{:.4f}'),
                ('Directional Acc.', 'directional_accuracy', '{:.2f}%'),
                ('Best Lag', 'best_lag', '{:.0f}'),
                ('Best Lag Corr', 'best_lag_corr', '{:.4f}')
            ]
            
            for label, key, fmt in metrics_to_show:
                row = [label]
                for horizon in sorted(self.results.keys()):
                    val = self.results[horizon].get(key, 0)
                    if '%' in fmt:
                        row.append(fmt.format(val))
                    else:
                        row.append(fmt.format(val))
                table.add_row(*row)
            
            console.print(table)
            console.print()
            
            # Interpretation panel
            self._print_interpretation()
            
        else:
            print("\n📊 Hasil Evaluasi Model")
            print("=" * 70)
            print(f"{'Metrik':<20}", end="")
            for horizon in sorted(self.results.keys()):
                print(f"{horizon} Hari".rjust(15), end="")
            print()
            print("-" * 70)
            
            metrics_to_show = [
                ('Samples', 'num_samples'),
                ('MSE', 'mse'),
                ('RMSE', 'rmse'),
                ('MAE', 'mae'),
                ('MAPE (%)', 'mape'),
                ('R² Score', 'r2'),
                ('Directional Acc.', 'directional_accuracy'),
                ('Best Lag', 'best_lag')
            ]
            
            for label, key in metrics_to_show:
                print(f"{label:<20}", end="")
                for horizon in sorted(self.results.keys()):
                    val = self.results[horizon].get(key, 0)
                    print(f"{val:>15.4f}", end="")
                print()
            print("=" * 70)
    
    def _print_interpretation(self):
        """Print interpretation of results"""
        if not self.results:
            return
        
        # Get average metrics
        avg_r2 = np.mean([r['r2'] for r in self.results.values()])
        avg_dir_acc = np.mean([r['directional_accuracy'] for r in self.results.values()])
        avg_mape = np.mean([r['mape'] for r in self.results.values()])
        
        # Check if best lag is 0 (good) or > 0 (possibly lagging)
        max_best_lag = max([r['best_lag'] for r in self.results.values()])
        
        interpretation = []
        
        # R² interpretation
        if avg_r2 > 0.7:
            interpretation.append(("✅", "R² tinggi: Model memiliki kemampuan prediksi yang baik", "bright_green"))
        elif avg_r2 > 0.4:
            interpretation.append(("⚠️", "R² sedang: Model cukup baik tapi masih bisa ditingkatkan", "bright_yellow"))
        else:
            interpretation.append(("❌", "R² rendah: Model mungkin hanya menghafal atau tidak efektif", "bright_red"))
        
        # Directional accuracy interpretation
        if avg_dir_acc > 60:
            interpretation.append(("✅", f"Directional Accuracy {avg_dir_acc:.1f}%: Model baik memprediksi arah harga", "bright_green"))
        elif avg_dir_acc > 52:
            interpretation.append(("⚠️", f"Directional Accuracy {avg_dir_acc:.1f}%: Sedikit lebih baik dari random", "bright_yellow"))
        else:
            interpretation.append(("❌", f"Directional Accuracy {avg_dir_acc:.1f}%: Sama dengan menebak random (50%)", "bright_red"))
        
        # Lag interpretation
        if max_best_lag > 1:
            interpretation.append(("⚠️", f"Best Lag = {max_best_lag}: Model mungkin 'mengikuti' harga dengan delay", "bright_yellow"))
        else:
            interpretation.append(("✅", "Best Lag = 0: Model tidak lag behind, prediksi real-time", "bright_green"))
        
        # MAPE interpretation  
        if avg_mape < 5:
            interpretation.append(("✅", f"MAPE {avg_mape:.1f}%: Error sangat rendah", "bright_green"))
        elif avg_mape < 10:
            interpretation.append(("⚠️", f"MAPE {avg_mape:.1f}%: Error cukup rendah", "bright_yellow"))
        else:
            interpretation.append(("❌", f"MAPE {avg_mape:.1f}%: Error tinggi", "bright_red"))
        
        # Overall verdict
        good_count = sum(1 for i in interpretation if i[0] == "✅")
        if good_count >= 3:
            verdict = ("🎯 KESIMPULAN: Model MEMPREDIKSI dengan baik, bukan sekadar menghafal", "bright_green bold")
        elif good_count >= 2:
            verdict = ("⚖️ KESIMPULAN: Model CUKUP BAIK, tapi ada ruang untuk improvement", "bright_yellow bold")
        else:
            verdict = ("🚫 KESIMPULAN: Model kemungkinan OVERFITTING atau menghafal data", "bright_red bold")
        
        if RICH_AVAILABLE:
            interp_text = "\n".join([f"{i[0]} {i[1]}" for i in interpretation])
            interp_text += f"\n\n{verdict[0]}"
            
            panel = Panel(
                interp_text,
                title="🔍 Interpretasi Hasil",
                border_style="bright_cyan",
                box=box.ROUNDED
            )
            console.print(panel)
        else:
            print("\n🔍 Interpretasi Hasil:")
            print("-" * 50)
            for icon, text, _ in interpretation:
                print(f"{icon} {text}")
            print()
            print(verdict[0])
            print("-" * 50)
    
    def plot_results(self, save_path=None):
        """Create visualization of predictions vs actuals"""
        if not self.results:
            return
        
        num_horizons = len(self.results)
        fig, axes = plt.subplots(2, num_horizons, figsize=(6*num_horizons, 10))
        
        if num_horizons == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (horizon, metrics) in enumerate(sorted(self.results.items())):
            # Top row: Predictions vs Actuals
            ax1 = axes[0, idx]
            sample_size = min(200, len(metrics['actuals']))
            ax1.plot(metrics['actuals'][:sample_size], label='Actual', color='blue', alpha=0.7)
            ax1.plot(metrics['predictions'][:sample_size], label='Predicted', color='red', alpha=0.7, linestyle='--')
            ax1.set_title(f'{horizon}-Day Forecast: Predictions vs Actual')
            ax1.set_xlabel('Sample')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add R² annotation
            ax1.text(0.02, 0.98, f"R² = {metrics['r2']:.4f}\nMAPE = {metrics['mape']:.2f}%", 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Bottom row: Error distribution
            ax2 = axes[1, idx]
            ax2.hist(metrics['errors'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_title(f'{horizon}-Day Forecast: Error Distribution')
            ax2.set_xlabel('Prediction Error')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Add mean error annotation
            mean_err = np.mean(metrics['errors'])
            std_err = np.std(metrics['errors'])
            ax2.text(0.98, 0.98, f"Mean = {mean_err:.2f}\nStd = {std_err:.2f}", 
                    transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{self.ticker} - Model Evaluation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print_success(f"Plot disimpan ke {save_path}")
        
        plt.close()
    
    def save_results_csv(self, save_path):
        """Save detailed results to CSV"""
        if not self.results:
            return
        
        rows = []
        for horizon, metrics in sorted(self.results.items()):
            row = {
                'Horizon (Days)': horizon,
                'Samples': metrics['num_samples'],
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAPE (%)': metrics['mape'],
                'R2 Score': metrics['r2'],
                'Directional Accuracy (%)': metrics['directional_accuracy'],
                'Best Lag': metrics['best_lag'],
                'Best Lag Correlation': metrics['best_lag_corr']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        
        if self.verbose:
            print_success(f"Hasil disimpan ke {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluasi kemampuan prediksi model PatchTST")
    
    parser.add_argument("--ticker", required=True, help="Kode saham (contoh: AAPL, BBCA.JK)")
    parser.add_argument("--total-years", type=int, default=10, help="Total tahun data (default: 10)")
    parser.add_argument("--train-years", type=int, default=5, help="Tahun untuk training (default: 5)")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback days (default: 60)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--horizons", type=str, default="20,30,60", help="Forecast horizons, comma-separated (default: 20,30,60)")
    parser.add_argument("--save-results", action="store_true", help="Simpan hasil ke file")
    parser.add_argument("--verbose", action="store_true", default=True, help="Tampilkan progress")
    
    return parser.parse_args()


def main():
    print_header()
    
    args = parse_args()
    
    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    
    print_info(f"Ticker: {args.ticker}")
    print_info(f"Data: {args.total_years} tahun, Training: {args.train_years} tahun")
    print_info(f"Lookback: {args.lookback} hari")
    print_info(f"Horizons: {horizons}")
    print()
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(args.ticker, lookback=args.lookback, verbose=args.verbose)
        
        # Load and split data
        if not evaluator.load_and_split_data(args.total_years, args.train_years):
            return 1
        
        # Train model
        if not evaluator.train_model(epochs=args.epochs):
            return 1
        
        # Run evaluation
        print()
        evaluator.run_full_evaluation(horizons=horizons)
        
        # Print results
        print()
        evaluator.print_results()
        
        # Save results if requested
        if args.save_results:
            if not os.path.exists("results"):
                os.makedirs("results")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"results/{args.ticker}_model_evaluation_{timestamp}.png"
            csv_path = f"results/{args.ticker}_model_evaluation_{timestamp}.csv"
            
            evaluator.plot_results(save_path=plot_path)
            evaluator.save_results_csv(csv_path)
        
        # Completion
        if RICH_AVAILABLE:
            console.print("\n[bright_green]✅ Evaluasi selesai![/bright_green]")
        else:
            print("\n✅ Evaluasi selesai!")
        
        return 0
        
    except Exception as e:
        print_error(f"Terjadi kesalahan: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
