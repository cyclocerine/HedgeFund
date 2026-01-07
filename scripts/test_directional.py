"""Quick test script for DirectionalEnsemble"""
import sys, os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.models.directional_model import DirectionalEnsemble, DirectionalFeatureEngineer
from src.data.preprocessor import DataPreprocessor
from datetime import datetime, timedelta
import numpy as np

# 6 years data: 5 train, 1 test
end = datetime.now()
start = end - timedelta(days=6*365)

print('Downloading BBCA.JK...')
prep = DataPreprocessor('BBCA.JK', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
prep.download_data()

fe = DirectionalFeatureEngineer()
data = fe.calculate_features(prep.data)

feature_cols = [c for c in fe.get_feature_columns() if c in data.columns]
data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

split = int(len(data) * 0.83)  # 5/6 for train
X_train = data[feature_cols].values[:split]
y_train = data['target'].values[:split]
X_test = data[feature_cols].values[split:]
y_test = data['target'].values[split:]

print(f'Train: {len(X_train)}, Test: {len(X_test)}')
print(f'Test UP ratio: {y_test.mean()*100:.1f}%')

model = DirectionalEnsemble(input_dim=len(feature_cols), lookback=30)
model.fit(X_train, y_train, epochs=30, verbose=0)

metrics, _ = model.evaluate(X_test, y_test)
individual = model.get_individual_accuracies(X_test, y_test)

print()
print('='*50)
print('FINAL RESULTS - BBCA.JK')
print('='*50)
acc = metrics['accuracy']
prec = metrics['precision']
rec = metrics['recall']
f1 = metrics['f1']
print(f'Ensemble Accuracy: {acc*100:.1f}%')
print(f'Precision: {prec*100:.1f}%')
print(f'Recall: {rec*100:.1f}%')
print(f'F1: {f1*100:.1f}%')
print()
print('Individual Models:')
for name, acc_val in sorted(individual.items(), key=lambda x: -x[1]):
    print(f'  {name}: {acc_val*100:.1f}%')
