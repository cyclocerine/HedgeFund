"""
Directional Classifier Model
============================

Model untuk prediksi arah harga (UP/DOWN) dengan:
- Klasifikasi binary alih-alih regresi
- Ensemble voting dari 4 model
- Enhanced features untuk momentum trading
- Horizon pendek (1-5 hari) untuk akurasi lebih tinggi
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import copy
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .patchtst_model import PositionalEncoding, EarlyStopping


class DirectionalFeatureEngineer:
    """Feature engineering untuk prediksi arah harga"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def calculate_features(self, df):
        """
        Calculate enhanced features untuk directional prediction.
        
        Args:
            df: DataFrame dengan kolom OHLCV
        
        Returns:
            DataFrame dengan features tambahan
        """
        data = df.copy()
        
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex by taking first level
            data.columns = data.columns.get_level_values(0)
        
        # Pastikan kolom Close ada
        if 'Close' not in data.columns:
            raise ValueError("DataFrame harus memiliki kolom 'Close'")
        
        # Convert to numeric and ensure proper Series
        close = pd.to_numeric(data['Close'], errors='coerce')
        high = pd.to_numeric(data['High'], errors='coerce') if 'High' in data.columns else close
        low = pd.to_numeric(data['Low'], errors='coerce') if 'Low' in data.columns else close
        volume = pd.to_numeric(data['Volume'], errors='coerce') if 'Volume' in data.columns else pd.Series(np.ones(len(close)), index=close.index)
        
        # Reset index to avoid alignment issues
        close = close.reset_index(drop=True)
        high = high.reset_index(drop=True)
        low = low.reset_index(drop=True)
        volume = volume.reset_index(drop=True)
        
        # Create new dataframe with proper index
        result = pd.DataFrame(index=range(len(close)))
        result['Close'] = close.values
        result['High'] = high.values
        result['Low'] = low.values
        result['Volume'] = volume.values
        
        # Use result columns as Series
        close_s = result['Close']
        high_s = result['High']
        low_s = result['Low']
        volume_s = result['Volume']
        
        # 1. Returns (target untuk classification)
        result['returns_1d'] = close_s.pct_change(1)
        result['returns_2d'] = close_s.pct_change(2)
        result['returns_3d'] = close_s.pct_change(3)
        result['returns_5d'] = close_s.pct_change(5)
        
        # 2. EMA (9, 20, 50)
        result['ema_9'] = close_s.ewm(span=9, adjust=False).mean()
        result['ema_20'] = close_s.ewm(span=20, adjust=False).mean()
        result['ema_50'] = close_s.ewm(span=50, adjust=False).mean()
        
        # EMA crossovers
        result['ema_9_20_cross'] = (result['ema_9'] > result['ema_20']).astype(int)
        result['ema_20_50_cross'] = (result['ema_20'] > result['ema_50']).astype(int)
        result['price_above_ema_20'] = (close_s > result['ema_20']).astype(int)
        result['price_above_ema_50'] = (close_s > result['ema_50']).astype(int)
        
        # 3. RSI
        delta = close_s.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        result['rsi'] = 100 - (100 / (1 + rs))
        result['rsi_overbought'] = (result['rsi'] > 70).astype(int)
        result['rsi_oversold'] = (result['rsi'] < 30).astype(int)
        
        # RSI divergence (price up but RSI down = bearish divergence)
        result['rsi_divergence'] = (np.sign(result['returns_5d']) != np.sign(result['rsi'].diff(5))).astype(int)
        
        # 4. MACD
        ema_12 = close_s.ewm(span=12, adjust=False).mean()
        ema_26 = close_s.ewm(span=26, adjust=False).mean()
        result['macd'] = ema_12 - ema_26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        result['macd_cross_up'] = ((result['macd'] > result['macd_signal']) & 
                                  (result['macd'].shift(1) <= result['macd_signal'].shift(1))).astype(int)
        result['macd_cross_down'] = ((result['macd'] < result['macd_signal']) & 
                                    (result['macd'].shift(1) >= result['macd_signal'].shift(1))).astype(int)
        result['macd_histogram_slope'] = result['macd_histogram'].diff()
        
        # 5. Bollinger Bands
        result['bb_middle'] = close_s.rolling(window=20).mean()
        bb_std = close_s.rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + (bb_std * 2)
        result['bb_lower'] = result['bb_middle'] - (bb_std * 2)
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / (result['bb_middle'] + 1e-10)
        result['bb_position'] = (close_s - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-10)
        result['bb_squeeze'] = (result['bb_width'] < result['bb_width'].rolling(20).mean()).astype(int)
        
        # 6. ADX (Average Directional Index)
        tr1 = (high_s - low_s).values
        tr2 = np.abs((high_s - close_s.shift(1)).values)
        tr3 = np.abs((low_s - close_s.shift(1)).values)
        tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3))
        atr = tr.rolling(window=14).mean()
        
        plus_dm = high_s.diff().copy()
        minus_dm = (-low_s.diff()).copy()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        result['adx'] = dx.rolling(14).mean()
        result['plus_di'] = plus_di.values
        result['minus_di'] = minus_di.values
        result['adx_trending'] = (result['adx'] > 25).astype(int)
        
        # 7. Volume features
        result['volume_sma'] = volume_s.rolling(window=20).mean()
        result['volume_ratio'] = volume_s / (result['volume_sma'] + 1e-10)
        result['volume_spike'] = (result['volume_ratio'] > 1.5).astype(int)
        
        # 8. Price momentum
        result['momentum_5'] = close_s / close_s.shift(5) - 1
        result['momentum_10'] = close_s / close_s.shift(10) - 1
        result['momentum_20'] = close_s / close_s.shift(20) - 1
        
        # 9. Volatility
        result['volatility_5'] = result['returns_1d'].rolling(5).std()
        result['volatility_20'] = result['returns_1d'].rolling(20).std()
        result['volatility_ratio'] = result['volatility_5'] / (result['volatility_20'] + 1e-10)
        
        # 10. Support/Resistance levels
        result['high_20'] = high_s.rolling(20).max()
        result['low_20'] = low_s.rolling(20).min()
        result['near_high'] = ((result['high_20'] - close_s) / (close_s + 1e-10) < 0.02).astype(int)
        result['near_low'] = ((close_s - result['low_20']) / (close_s + 1e-10) < 0.02).astype(int)
        
        # Drop NaN rows
        result = result.dropna()
        
        return result
    
    def get_feature_columns(self):
        """Return list of feature columns untuk model"""
        return [
            # Returns
            'returns_1d', 'returns_2d', 'returns_3d', 'returns_5d',
            # EMA
            'ema_9_20_cross', 'ema_20_50_cross', 'price_above_ema_20', 'price_above_ema_50',
            # RSI
            'rsi', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
            # MACD
            'macd_histogram', 'macd_cross_up', 'macd_cross_down', 'macd_histogram_slope',
            # Bollinger
            'bb_position', 'bb_width', 'bb_squeeze',
            # ADX
            'adx', 'plus_di', 'minus_di', 'adx_trending',
            # Volume
            'volume_ratio', 'volume_spike',
            # Momentum
            'momentum_5', 'momentum_10', 'momentum_20',
            # Volatility
            'volatility_5', 'volatility_20', 'volatility_ratio',
            # Support/Resistance
            'near_high', 'near_low'
        ]


class TransformerClassifier(nn.Module):
    """Transformer untuk binary classification"""
    
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, 
            dim_feedforward=d_model*2, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM untuk binary classification"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Last timestep
        return self.classifier(out)


class DirectionalEnsemble:
    """
    Ensemble classifier untuk prediksi arah harga.
    Menggunakan voting dari 4 model: Transformer, LSTM, XGBoost, RandomForest
    """
    
    def __init__(self, input_dim, lookback=20, device=None):
        self.input_dim = input_dim
        self.lookback = lookback
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_engineer = DirectionalFeatureEngineer()
        self.scaler = StandardScaler()
        
        # Initialize models
        self.models = {}
        self._init_models()
        
        # Voting weights (can be optimized)
        self.weights = {
            'transformer': 0.30,
            'lstm': 0.30,
            'xgboost': 0.25,
            'random_forest': 0.15
        }
        
        # Training config
        self.threshold = 0.5  # Classification threshold
    
    def _init_models(self):
        """Initialize all ensemble models"""
        # PyTorch models
        self.models['transformer'] = TransformerClassifier(
            input_dim=self.input_dim
        ).to(self.device)
        
        self.models['lstm'] = LSTMClassifier(
            input_dim=self.input_dim
        ).to(self.device)
        
        # Sklearn models
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )
        else:
            self.models['xgboost'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    
    def prepare_data(self, df, horizon=1):
        """
        Prepare data untuk training.
        
        Args:
            df: DataFrame dengan OHLCV data
            horizon: Prediction horizon (1-5 days)
        
        Returns:
            X, y arrays untuk training
        """
        # Calculate features
        data = self.feature_engineer.calculate_features(df)
        
        # Create target: 1 if price goes UP in next 'horizon' days, 0 otherwise
        data['target'] = (data['Close'].shift(-horizon) > data['Close']).astype(int)
        
        # Remove rows without target
        data = data.dropna()
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_columns()
        available_cols = [c for c in feature_cols if c in data.columns]
        
        X = data[available_cols].values
        y = data['target'].values
        
        return X, y, data.index
    
    def create_sequences(self, X, y):
        """Create sequences untuk LSTM/Transformer"""
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i-self.lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X, y, epochs=50, batch_size=32, verbose=1):
        """
        Train semua model dalam ensemble.
        """
        print("=" * 60)
        print("TRAINING DIRECTIONAL ENSEMBLE")
        print("=" * 60)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences for deep learning models
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        # Train/val split
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Flatten for tree-based models
        X_flat_train = X_train.reshape(X_train.shape[0], -1)
        X_flat_val = X_val.reshape(X_val.shape[0], -1)
        
        # 1. Train Transformer
        print("\n[1/4] Training Transformer...")
        self._train_pytorch_model(
            self.models['transformer'],
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )
        
        # 2. Train LSTM
        print("\n[2/4] Training LSTM...")
        self._train_pytorch_model(
            self.models['lstm'],
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )
        
        # 3. Train XGBoost
        print("\n[3/4] Training XGBoost...")
        self.models['xgboost'].fit(X_flat_train, y_train)
        xgb_acc = accuracy_score(y_val, self.models['xgboost'].predict(X_flat_val))
        print(f"XGBoost Validation Accuracy: {xgb_acc:.4f}")
        
        # 4. Train Random Forest
        print("\n[4/4] Training Random Forest...")
        self.models['random_forest'].fit(X_flat_train, y_train)
        rf_acc = accuracy_score(y_val, self.models['random_forest'].predict(X_flat_val))
        print(f"Random Forest Validation Accuracy: {rf_acc:.4f}")
        
        print("\n" + "=" * 60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("=" * 60)
        
        # Optimize threshold on validation set
        self._optimize_threshold(X_val, y_val, X_flat_val)
    
    def _train_pytorch_model(self, model, X_train, y_train, X_val, y_val,
                             epochs=50, batch_size=32, verbose=1):
        """Train a PyTorch model"""
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        
        n = len(X_train_t)
        
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            X_train_t = X_train_t[perm]
            y_train_t = y_train_t[perm]
            
            total_loss = 0
            for i in range(0, n, batch_size):
                xb = X_train_t[i:i+batch_size]
                yb = y_train_t[i:i+batch_size]
                
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
                val_acc = ((val_pred > 0.5) == y_val_t).float().mean().item()
            
            scheduler.step(val_loss)
            
            if early_stopping(val_loss, model):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                early_stopping.restore_weights(model)
                break
            
            if verbose and (epoch % 10 == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def _optimize_threshold(self, X_val, y_val, X_flat_val):
        """Optimize classification threshold"""
        probs = self._get_ensemble_probs(X_val, X_flat_val)
        
        best_acc = 0
        best_threshold = 0.5
        
        for thresh in np.arange(0.3, 0.7, 0.05):
            preds = (probs > thresh).astype(int)
            acc = accuracy_score(y_val, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = thresh
        
        self.threshold = best_threshold
        print(f"\nOptimized threshold: {best_threshold:.2f} (accuracy: {best_acc:.4f})")
    
    def _get_ensemble_probs(self, X_seq, X_flat):
        """Get probability predictions from all models"""
        probs = np.zeros(len(X_seq))
        
        # Transformer
        self.models['transformer'].eval()
        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            trans_probs = self.models['transformer'](X_t).cpu().numpy().flatten()
        probs += self.weights['transformer'] * trans_probs
        
        # LSTM
        self.models['lstm'].eval()
        with torch.no_grad():
            lstm_probs = self.models['lstm'](X_t).cpu().numpy().flatten()
        probs += self.weights['lstm'] * lstm_probs
        
        # XGBoost
        if hasattr(self.models['xgboost'], 'predict_proba'):
            xgb_probs = self.models['xgboost'].predict_proba(X_flat)[:, 1]
        else:
            xgb_probs = self.models['xgboost'].predict(X_flat)
        probs += self.weights['xgboost'] * xgb_probs
        
        # Random Forest
        rf_probs = self.models['random_forest'].predict_proba(X_flat)[:, 1]
        probs += self.weights['random_forest'] * rf_probs
        
        return probs
    
    def predict(self, X):
        """
        Predict direction: 1 = UP, 0 = DOWN
        """
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        
        probs = self._get_ensemble_probs(X_seq, X_flat)
        return (probs > self.threshold).astype(int), probs
    
    def predict_single(self, X_seq):
        """Predict single sequence"""
        X_flat = X_seq.reshape(1, -1)
        X_seq = X_seq.reshape(1, *X_seq.shape)
        
        probs = self._get_ensemble_probs(X_seq, X_flat)
        return int(probs[0] > self.threshold), probs[0]
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        
        probs = self._get_ensemble_probs(X_seq, X_flat)
        preds = (probs > self.threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_seq, preds),
            'precision': precision_score(y_seq, preds, zero_division=0),
            'recall': recall_score(y_seq, preds, zero_division=0),
            'f1': f1_score(y_seq, preds, zero_division=0)
        }
        
        return metrics, classification_report(y_seq, preds)
    
    def get_individual_accuracies(self, X, y):
        """Get accuracy for each individual model"""
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        
        accuracies = {}
        
        # Transformer
        self.models['transformer'].eval()
        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            trans_preds = (self.models['transformer'](X_t).cpu().numpy().flatten() > 0.5).astype(int)
        accuracies['transformer'] = accuracy_score(y_seq, trans_preds)
        
        # LSTM
        self.models['lstm'].eval()
        with torch.no_grad():
            lstm_preds = (self.models['lstm'](X_t).cpu().numpy().flatten() > 0.5).astype(int)
        accuracies['lstm'] = accuracy_score(y_seq, lstm_preds)
        
        # XGBoost
        xgb_preds = self.models['xgboost'].predict(X_flat)
        accuracies['xgboost'] = accuracy_score(y_seq, xgb_preds)
        
        # Random Forest
        rf_preds = self.models['random_forest'].predict(X_flat)
        accuracies['random_forest'] = accuracy_score(y_seq, rf_preds)
        
        return accuracies
    
    def save(self, path_prefix):
        """Save all models"""
        import pickle
        
        # Save PyTorch models
        torch.save(self.models['transformer'].state_dict(), f"{path_prefix}_transformer.pt")
        torch.save(self.models['lstm'].state_dict(), f"{path_prefix}_lstm.pt")
        
        # Save sklearn models
        with open(f"{path_prefix}_xgboost.pkl", 'wb') as f:
            pickle.dump(self.models['xgboost'], f)
        with open(f"{path_prefix}_rf.pkl", 'wb') as f:
            pickle.dump(self.models['random_forest'], f)
        
        # Save config
        config = {
            'weights': self.weights,
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'lookback': self.lookback,
            'scaler': self.scaler
        }
        with open(f"{path_prefix}_config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Models saved to {path_prefix}_*")
    
    def load(self, path_prefix):
        """Load all models"""
        import pickle
        
        # Load config
        with open(f"{path_prefix}_config.pkl", 'rb') as f:
            config = pickle.load(f)
        
        self.weights = config['weights']
        self.threshold = config['threshold']
        self.scaler = config['scaler']
        
        # Load PyTorch models
        self.models['transformer'].load_state_dict(
            torch.load(f"{path_prefix}_transformer.pt", map_location=self.device)
        )
        self.models['lstm'].load_state_dict(
            torch.load(f"{path_prefix}_lstm.pt", map_location=self.device)
        )
        
        # Load sklearn models
        with open(f"{path_prefix}_xgboost.pkl", 'rb') as f:
            self.models['xgboost'] = pickle.load(f)
        with open(f"{path_prefix}_rf.pkl", 'rb') as f:
            self.models['random_forest'] = pickle.load(f)
        
        print(f"Models loaded from {path_prefix}")
