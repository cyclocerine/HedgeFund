import traceback
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.predictor import StockPredictor

try:
    predictor = StockPredictor(
        ticker='BBNI.JK',
        start_date='2020-01-01',
        end_date='2026-03-27',
        lookback=60,
        forecast_days=30,
        model_type='plstm',
        tune_hyperparameters=False
    )
    predictor.prepare_data()
    predictor.train_model()
    y_true, y_pred, forecast = predictor.predict()
    print("Forecast length:", len(forecast))
    print("Forecast values:", forecast)
except Exception as e:
    with open("crash_log.txt", "w") as f:
        traceback.print_exc(file=f)
    print("Crashed. See crash_log.txt")
    sys.exit(1)
