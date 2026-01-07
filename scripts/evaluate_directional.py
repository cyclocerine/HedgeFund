#!/usr/bin/env python
"""
Evaluate Directional Model
==========================

Script untuk mengevaluasi DirectionalClassifier dengan target 85%+ accuracy.
"""

import argparse
from datetime import datetime, timedelta
import sys
import os

os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.models.directional_model import DirectionalEnsemble, DirectionalFeatureEngineer
from src.data.preprocessor import DataPreprocessor

console = Console() if RICH_AVAILABLE else None


def print_header():
    if RICH_AVAILABLE:
        header = """
╔════════════════════════════════════════════════════════════════╗
║  DIRECTIONAL ACCURACY EVALUATION                               ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║
║  Target: 85%+ | Ensemble Voting | Binary Classification        ║
╚════════════════════════════════════════════════════════════════╝
"""
        console.print(header, style="bright_cyan")
    else:
        print("=" * 60)
        print("  DIRECTIONAL ACCURACY EVALUATION")
        print("  Target: 85%+ | Ensemble Voting")
        print("=" * 60)


def print_info(msg):
    if RICH_AVAILABLE:
        console.print(f"[INFO] {msg}", style="bright_blue")
    else:
        print(f"[INFO] {msg}")


def print_success(msg):
    if RICH_AVAILABLE:
        console.print(f"[OK] {msg}", style="bright_green")
    else:
        print(f"[OK] {msg}")


def print_error(msg):
    if RICH_AVAILABLE:
        console.print(f"[ERROR] {msg}", style="bold red")
    else:
        print(f"[ERROR] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Directional Classifier")
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., BBCA.JK)")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in days (default: 1)")
    parser.add_argument("--lookback", type=int, default=20, help="Lookback days (default: 20)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--train-years", type=int, default=4, help="Years for training (default: 4)")
    parser.add_argument("--test-years", type=int, default=1, help="Years for testing (default: 1)")
    parser.add_argument("--save-model", action="store_true", help="Save trained model")
    
    args = parser.parse_args()
    
    print_header()
    
    print_info(f"Ticker: {args.ticker}")
    print_info(f"Prediction Horizon: {args.horizon} day(s)")
    print_info(f"Lookback: {args.lookback} days")
    print_info(f"Training: {args.train_years} years, Testing: {args.test_years} years")
    print()
    
    try:
        # Download data
        total_years = args.train_years + args.test_years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_years * 365)
        
        print_info("Downloading data...")
        preprocessor = DataPreprocessor(
            args.ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not preprocessor.download_data():
            print_error("Failed to download data")
            return 1
        
        data = preprocessor.data
        print_success(f"Downloaded {len(data)} data points")
        
        # Prepare features
        print_info("Engineering features...")
        feature_eng = DirectionalFeatureEngineer()
        data_with_features = feature_eng.calculate_features(data)
        print_success(f"Created {len(feature_eng.get_feature_columns())} features")
        
        # Split train/test
        split_idx = int(len(data_with_features) * (args.train_years / total_years))
        train_data = data_with_features.iloc[:split_idx]
        test_data = data_with_features.iloc[split_idx:]
        
        print_info(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # Prepare training data
        feature_cols = feature_eng.get_feature_columns()
        available_cols = [c for c in feature_cols if c in train_data.columns]
        
        # Create target
        train_data = train_data.copy()
        test_data = test_data.copy()
        train_data['target'] = (train_data['Close'].shift(-args.horizon) > train_data['Close']).astype(int)
        test_data['target'] = (test_data['Close'].shift(-args.horizon) > test_data['Close']).astype(int)
        
        train_data = train_data.dropna()
        test_data = test_data.dropna()
        
        X_train = train_data[available_cols].values
        y_train = train_data['target'].values
        X_test = test_data[available_cols].values
        y_test = test_data['target'].values
        
        print_info(f"Features used: {len(available_cols)}")
        print_info(f"Train class distribution: UP={y_train.sum()}/{len(y_train)} ({100*y_train.mean():.1f}%)")
        print_info(f"Test class distribution: UP={y_test.sum()}/{len(y_test)} ({100*y_test.mean():.1f}%)")
        print()
        
        # Initialize and train model
        model = DirectionalEnsemble(
            input_dim=len(available_cols),
            lookback=args.lookback
        )
        
        model.fit(X_train, y_train, epochs=args.epochs, verbose=1)
        
        # Evaluate on test set
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)
        
        metrics, report = model.evaluate(X_test, y_test)
        
        # Print results
        if RICH_AVAILABLE:
            table = Table(title="📊 Directional Accuracy Results", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")
            table.add_column("Target", style="yellow", justify="right")
            table.add_column("Status", justify="center")
            
            for metric, value in metrics.items():
                target = 0.85 if metric == 'accuracy' else 0.80
                status = "✅" if value >= target else "⚠️" if value >= 0.6 else "❌"
                table.add_row(
                    metric.capitalize(),
                    f"{value:.4f} ({value*100:.1f}%)",
                    f"{target*100:.0f}%",
                    status
                )
            
            console.print(table)
            console.print()
            
            # Individual model accuracies
            individual = model.get_individual_accuracies(X_test, y_test)
            
            ind_table = Table(title="Individual Model Accuracies", box=box.SIMPLE)
            ind_table.add_column("Model", style="cyan")
            ind_table.add_column("Accuracy", style="green", justify="right")
            
            for name, acc in individual.items():
                ind_table.add_row(name.capitalize(), f"{acc:.4f} ({acc*100:.1f}%)")
            
            console.print(ind_table)
            
            # Interpretation
            acc = metrics['accuracy']
            if acc >= 0.85:
                verdict = "🎯 TARGET TERCAPAI! Accuracy >= 85%"
                style = "bright_green bold"
            elif acc >= 0.70:
                verdict = "⚠️ CUKUP BAIK - Accuracy 70-85%"
                style = "bright_yellow bold"
            elif acc >= 0.55:
                verdict = "⚠️ MASIH IMPROVEMENT - Accuracy 55-70%"
                style = "yellow"
            else:
                verdict = "❌ PERLU PERBAIKAN - Accuracy < 55%"
                style = "bright_red bold"
            
            console.print(Panel(verdict, border_style=style))
            
        else:
            print("\nResults:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f} ({value*100:.1f}%)")
            
            print("\nClassification Report:")
            print(report)
        
        # Save model if requested
        if args.save_model:
            if not os.path.exists("models"):
                os.makedirs("models")
            path = f"models/{args.ticker}_directional_h{args.horizon}"
            model.save(path)
            print_success(f"Model saved to {path}_*")
        
        print_success("Evaluation complete!")
        
        return 0
        
    except Exception as e:
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
