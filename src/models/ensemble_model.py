"""
Ensemble Model
==============

Ensemble predictor yang menggabungkan:
- PatchTST: Menangkap pattern global jangka panjang
- BiLSTM: Menangkap pattern sequential jangka pendek
- XGBoost: Robust terhadap outlier, feature engineering

Metode ensemble:
- Weighted averaging
- Stacking dengan meta-learner
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import copy
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. XGBoost ensemble will be disabled.")

from .patchtst_model import ImprovedPatchTSTWrapper, EarlyStopping


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM untuk time series forecasting"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, 
                 forecast_horizons=[20, 30, 60, 100]):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizons = forecast_horizons
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Multi-horizon heads
        self.heads = nn.ModuleDict()
        for h in forecast_horizons:
            self.heads[f'h_{h}'] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, h)
            )
        
        # Single step head
        self.single_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, horizon=None):
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            horizon: Forecast horizon. If None, returns single step prediction
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        out = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        out = self.layer_norm(out)
        
        if horizon is None:
            return self.single_head(out)
        elif horizon in self.forecast_horizons:
            return self.heads[f'h_{horizon}'](out)
        else:
            raise ValueError(f"Horizon {horizon} not in {self.forecast_horizons}")


class BiLSTMWrapper:
    """Wrapper untuk BiLSTM dengan training utilities"""
    
    def __init__(self, input_dim, device=None, progress_callback=None,
                 forecast_horizons=[20, 30, 60, 100], **kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.forecast_horizons = forecast_horizons
        
        hidden_dim = kwargs.pop('hidden_dim', 128)
        num_layers = kwargs.pop('num_layers', 2)
        dropout = kwargs.pop('dropout', 0.2)
        lr = kwargs.pop('lr', 1e-3)
        
        self.model = BiLSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizons=forecast_horizons
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        self.progress_callback = progress_callback
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32,
            verbose=1, early_stopping_patience=10, horizon=None):
        """Train BiLSTM model"""
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        
        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            if y_val.ndim == 1:
                y_val = y_val.unsqueeze(1)
        
        n = X_train.shape[0]
        early_stopping = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)
        
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, n, batch_size):
                xb = X_shuffled[i:i+batch_size]
                yb = y_shuffled[i:i+batch_size]
                
                self.optimizer.zero_grad()
                out = self.model(xb, horizon=horizon)
                
                if horizon is not None and yb.shape[1] != out.shape[1]:
                    yb = yb[:, :out.shape[1]]
                
                loss = self.criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validation
            val_loss = avg_loss
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val, horizon=horizon)
                    if horizon is not None and y_val.shape[1] != val_out.shape[1]:
                        y_val_adj = y_val[:, :val_out.shape[1]]
                    else:
                        y_val_adj = y_val
                    val_loss = self.criterion(val_out, y_val_adj).item()
            
            self.scheduler.step(val_loss)
            
            if early_stopping(val_loss, self.model):
                if verbose:
                    print(f"BiLSTM Early stopping at epoch {epoch+1}")
                early_stopping.restore_weights(self.model)
                break
            
            if self.progress_callback:
                progress = int((epoch + 1) / epochs * 100)
                self.progress_callback(progress, f"BiLSTM Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            elif verbose and (epoch % 10 == 0 or epoch == epochs-1):
                print(f"BiLSTM Epoch {epoch+1}/{epochs}, Train: {avg_loss:.4f}, Val: {val_loss:.4f}")
    
    def predict(self, X, horizon=None, batch_size=1024):
        """Predict dengan batched inference"""
        self.model.eval()
        
        if len(X) <= batch_size:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(X_tensor, horizon=horizon)
            return out.cpu().numpy()
        
        all_preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            X_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(X_tensor, horizon=horizon)
            all_preds.append(out.cpu().numpy())
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return np.concatenate(all_preds)


class XGBoostWrapper:
    """XGBoost wrapper untuk time series forecasting"""
    
    def __init__(self, forecast_horizons=[20, 30, 60, 100], **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
        
        self.forecast_horizons = forecast_horizons
        self.models = {}
        
        # XGBoost parameters
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'n_estimators': kwargs.get('n_estimators', 100),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.scaler = StandardScaler()
    
    def _flatten_sequences(self, X):
        """Flatten 3D sequences to 2D for XGBoost"""
        # X shape: (n_samples, seq_len, n_features)
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, horizon=None, verbose=1, **kwargs):
        """Train XGBoost models"""
        # Flatten sequences
        X_flat = self._flatten_sequences(X_train)
        X_flat = self.scaler.fit_transform(X_flat)
        
        if horizon is not None:
            # Train single model for specific horizon
            horizons_to_train = [horizon]
        else:
            horizons_to_train = self.forecast_horizons
        
        for h in horizons_to_train:
            if verbose:
                print(f"Training XGBoost for {h}-day horizon...")
            
            # For multi-step, we train multiple models (one per day)
            # Or use multi-output regression
            self.models[h] = xgb.XGBRegressor(**self.params)
            
            # Target: if y_train is just single values, use them
            # Otherwise, use appropriate slice
            if y_train.ndim == 1:
                y_target = y_train
            else:
                y_target = y_train[:, 0] if y_train.shape[1] >= 1 else y_train.flatten()
            
            self.models[h].fit(X_flat, y_target)
    
    def predict(self, X, horizon=None):
        """Predict with XGBoost"""
        X_flat = self._flatten_sequences(X)
        X_flat = self.scaler.transform(X_flat)
        
        if horizon is None:
            # Return single step prediction using first available model
            h = self.forecast_horizons[0] if self.forecast_horizons else list(self.models.keys())[0]
            return self.models[h].predict(X_flat).reshape(-1, 1)
        
        if horizon in self.models:
            pred = self.models[horizon].predict(X_flat)
            return pred.reshape(-1, 1)
        else:
            raise ValueError(f"No model trained for horizon {horizon}")


class EnsemblePredictor:
    """
    Ensemble Predictor yang menggabungkan multiple models:
    - PatchTST
    - BiLSTM
    - XGBoost (optional)
    
    Metode ensemble:
    - weighted_average: Rata-rata tertimbang berdasarkan weights
    - stacking: Meta-learner untuk kombinasi optimal
    """
    
    def __init__(self, input_dim, forecast_horizons=[20, 30, 60, 100],
                 use_xgboost=True, device=None, progress_callback=None):
        self.input_dim = input_dim
        self.forecast_horizons = forecast_horizons
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.progress_callback = progress_callback
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        
        # Initialize models
        self.models = {}
        
        # PatchTST
        self.models['patchtst'] = ImprovedPatchTSTWrapper(
            input_dim=input_dim,
            forecast_horizons=forecast_horizons,
            device=self.device,
            progress_callback=progress_callback
        )
        
        # BiLSTM
        self.models['bilstm'] = BiLSTMWrapper(
            input_dim=input_dim,
            forecast_horizons=forecast_horizons,
            device=self.device,
            progress_callback=progress_callback
        )
        
        # XGBoost (optional)
        if self.use_xgboost:
            self.models['xgboost'] = XGBoostWrapper(forecast_horizons=forecast_horizons)
        
        # Default weights
        if self.use_xgboost:
            self.weights = {'patchtst': 0.4, 'bilstm': 0.35, 'xgboost': 0.25}
        else:
            self.weights = {'patchtst': 0.55, 'bilstm': 0.45}
        
        # Meta-learner for stacking
        self.meta_learner = None
        self.ensemble_method = 'weighted_average'
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32,
            verbose=1, early_stopping_patience=10, horizon=None):
        """
        Train all models in the ensemble.
        
        Args:
            horizon: If None, train for single-step. Otherwise train for specific horizon.
        """
        print("=" * 60)
        print("TRAINING ENSEMBLE MODEL")
        print("=" * 60)
        
        # Train PatchTST
        print("\n[1/3] Training PatchTST...")
        self.models['patchtst'].fit(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose, early_stopping_patience=early_stopping_patience,
            horizon=horizon
        )
        
        # Train BiLSTM
        print("\n[2/3] Training BiLSTM...")
        self.models['bilstm'].fit(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose, early_stopping_patience=early_stopping_patience,
            horizon=horizon
        )
        
        # Train XGBoost
        if self.use_xgboost:
            print("\n[3/3] Training XGBoost...")
            self.models['xgboost'].fit(
                X_train, y_train, X_val, y_val,
                horizon=horizon, verbose=verbose
            )
        
        print("\n" + "=" * 60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("=" * 60)
    
    def fit_stacking(self, X_train, y_train, X_val, y_val, horizon=None):
        """
        Train ensemble dengan stacking method.
        Uses validation predictions as features for meta-learner.
        """
        # First, train all base models
        self.fit(X_train, y_train, X_val, y_val, horizon=horizon)
        
        # Get predictions on validation set
        val_preds = []
        for name, model in self.models.items():
            pred = model.predict(X_val, horizon=horizon)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            val_preds.append(pred)
        
        # Stack predictions
        stacked_preds = np.hstack(val_preds)
        
        # Train meta-learner (Ridge regression)
        self.meta_learner = Ridge(alpha=1.0)
        if y_val.ndim > 1:
            y_target = y_val[:, 0] if y_val.shape[1] > 0 else y_val.flatten()
        else:
            y_target = y_val
        
        self.meta_learner.fit(stacked_preds, y_target)
        self.ensemble_method = 'stacking'
        
        print("Stacking meta-learner trained successfully!")
    
    def predict(self, X, horizon=None, return_individual=False):
        """
        Make predictions using ensemble.
        
        Args:
            X: Input data
            horizon: Forecast horizon
            return_individual: If True, return predictions from each model
        """
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X, horizon=horizon)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            predictions[name] = pred
        
        if return_individual:
            return predictions
        
        # Combine predictions
        if self.ensemble_method == 'stacking' and self.meta_learner is not None:
            stacked = np.hstack([predictions[name] for name in self.models.keys()])
            final_pred = self.meta_learner.predict(stacked)
            return final_pred.reshape(-1, 1)
        else:
            # Weighted average
            final_pred = sum(
                self.weights[name] * predictions[name] 
                for name in predictions.keys()
            )
            return final_pred
    
    def set_weights(self, weights):
        """Set custom weights for weighted averaging"""
        assert sum(weights.values()) == 1.0, "Weights must sum to 1.0"
        self.weights = weights
    
    def optimize_weights(self, X_val, y_val, horizon=None):
        """
        Optimize ensemble weights using validation data.
        Uses grid search to find optimal weights.
        """
        predictions = self.predict(X_val, horizon=horizon, return_individual=True)
        
        if y_val.ndim > 1:
            y_target = y_val[:, 0]
        else:
            y_target = y_val
        
        best_mse = float('inf')
        best_weights = self.weights.copy()
        
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        # Grid search over weight combinations
        from itertools import product
        weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        for combo in product(weight_options, repeat=n_models):
            if abs(sum(combo) - 1.0) > 0.001:
                continue
            
            weights = dict(zip(model_names, combo))
            combined = sum(weights[name] * predictions[name].flatten() for name in model_names)
            mse = np.mean((combined - y_target) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_weights = weights.copy()
        
        print(f"Optimized weights: {best_weights}")
        print(f"Validation MSE: {best_mse:.4f}")
        
        self.weights = best_weights
        return best_weights, best_mse
    
    def save(self, path_prefix):
        """Save all models"""
        import pickle
        
        # Save PatchTST
        self.models['patchtst'].save(f"{path_prefix}_patchtst.pt")
        
        # Save BiLSTM
        torch.save(self.models['bilstm'].model.state_dict(), f"{path_prefix}_bilstm.pt")
        
        # Save XGBoost
        if self.use_xgboost:
            with open(f"{path_prefix}_xgboost.pkl", 'wb') as f:
                pickle.dump(self.models['xgboost'], f)
        
        # Save meta-learner and config
        config = {
            'weights': self.weights,
            'ensemble_method': self.ensemble_method,
            'forecast_horizons': self.forecast_horizons,
            'use_xgboost': self.use_xgboost
        }
        if self.meta_learner is not None:
            config['meta_learner'] = self.meta_learner
        
        with open(f"{path_prefix}_config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Ensemble saved to {path_prefix}_*.pt/pkl")
    
    def load(self, path_prefix, input_dim):
        """Load all models"""
        import pickle
        
        # Load config
        with open(f"{path_prefix}_config.pkl", 'rb') as f:
            config = pickle.load(f)
        
        self.weights = config['weights']
        self.ensemble_method = config['ensemble_method']
        self.forecast_horizons = config['forecast_horizons']
        self.use_xgboost = config['use_xgboost']
        self.meta_learner = config.get('meta_learner')
        
        # Reinitialize and load models
        self.models['patchtst'] = ImprovedPatchTSTWrapper(
            input_dim=input_dim,
            forecast_horizons=self.forecast_horizons,
            device=self.device
        )
        self.models['patchtst'].load(f"{path_prefix}_patchtst.pt")
        
        self.models['bilstm'] = BiLSTMWrapper(
            input_dim=input_dim,
            forecast_horizons=self.forecast_horizons,
            device=self.device
        )
        self.models['bilstm'].model.load_state_dict(
            torch.load(f"{path_prefix}_bilstm.pt", map_location=self.device)
        )
        
        if self.use_xgboost:
            with open(f"{path_prefix}_xgboost.pkl", 'rb') as f:
                self.models['xgboost'] = pickle.load(f)
        
        print(f"Ensemble loaded from {path_prefix}")
