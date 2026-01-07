"""Predict BBCA.JK direction for tomorrow"""
import sys, os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.models.directional_model import DirectionalEnsemble, DirectionalFeatureEngineer
from src.data.preprocessor import DataPreprocessor
from datetime import datetime, timedelta
import numpy as np

end = datetime.now()
start = end - timedelta(days=6*365)

print('='*50)
print('PREDIKSI ARAH HARGA BBCA.JK')
print('='*50)
print()

print('Downloading data...')
prep = DataPreprocessor('BBCA.JK', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
prep.download_data()

fe = DirectionalFeatureEngineer()
data = fe.calculate_features(prep.data)

feature_cols = [c for c in fe.get_feature_columns() if c in data.columns]
data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# Use all data for training, predict last day
X_all = data[feature_cols].values
y_all = data['target'].values

# Train on all except last 30 days
split = len(X_all) - 30
X_train, y_train = X_all[:split], y_all[:split]

print(f'Training on {len(X_train)} samples...')
model = DirectionalEnsemble(input_dim=len(feature_cols), lookback=30)
model.fit(X_train, y_train, epochs=30, verbose=0)

# Predict latest
X_latest = X_all[-31:]
preds, probs = model.predict(X_latest)

# Get last close price
last_close = data['Close'].iloc[-1]
last_date = prep.data.index[-1]

print()
print('='*50)
print('HASIL PREDIKSI BBCA.JK')
print('='*50)
print(f'Tanggal terakhir: {last_date.strftime("%Y-%m-%d")}')
print(f'Harga terakhir: Rp {last_close:,.0f}')
print()

if preds[-1] == 1:
    direction = 'NAIK'
    emoji = '📈'
else:
    direction = 'TURUN'
    emoji = '📉'

confidence = probs[-1] * 100
print(f'{emoji} Prediksi besok: {direction}')
print(f'Confidence: {confidence:.1f}%')
print()
print('Note: Threshold = 50% (>50% = NAIK, <50% = TURUN)')
