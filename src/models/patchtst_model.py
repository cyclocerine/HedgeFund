"""
Improved PatchTST Model
=======================

Model PatchTST dengan improvement:
- Positional Encoding untuk Transformer
- Early Stopping untuk mencegah overfitting
- Direct Multi-Step Forecasting
- Better architecture dengan Layer Normalization

Feature Engineering Integration (v2.0):
- Helper function prepare_patchtst_features() untuk menyiapkan input stationary
- Support signal scoring (MACD, Stoch RSI, BB, Volume)
- Log Returns dan Rolling Z-Score untuk stationarity
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import copy
import optuna

# Import feature engineering (optional)
try:
    from src.data.feature_engineering import prepare_patchtst_features as _prepare_features
    HAS_FEATURE_ENGINEERING = True
except ImportError:
    HAS_FEATURE_ENGINEERING = False
    _prepare_features = None


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding untuk Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EarlyStopping:
    """Early Stopping untuk mencegah overfitting"""
    
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True):
        """
        Args:
            patience: Jumlah epoch tanpa improvement sebelum berhenti
            min_delta: Minimum change untuk dianggap sebagai improvement
            restore_best_weights: Apakah mengembalikan weights terbaik
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
    
    def __call__(self, val_loss, model):
        """
        Returns:
            True jika harus berhenti, False jika lanjut training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False
    
    def restore_weights(self, model):
        """Restore best weights ke model"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ImprovedPatchTST(nn.Module):
    """
    Improved PatchTST dengan:
    - Positional Encoding
    - Layer Normalization
    - Multi-horizon output untuk direct forecasting
    """
    
    def __init__(self, input_dim, patch_len=16, stride=8, d_model=128, n_heads=4, 
                 n_layers=2, dropout=0.1, forecast_horizons=[20, 30, 60, 100]):
        super().__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.input_dim = input_dim
        self.forecast_horizons = forecast_horizons
        
        # Patch embedding
        self.patch_embed = nn.Conv1d(input_dim, d_model, kernel_size=patch_len, stride=stride)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Input layer normalization
        self.input_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            dropout=dropout, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer normalization
        self.output_norm = nn.LayerNorm(d_model)
        
        # Multi-horizon heads untuk direct forecasting
        self.heads = nn.ModuleDict()
        for h in forecast_horizons:
            self.heads[f'h_{h}'] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, h)  # Output h days directly
            )
        
        # Single step head untuk backward compatibility
        self.single_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, horizon=None):
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            horizon: Forecast horizon. If None, returns single step prediction
        """
        # Patch embedding: (batch, input_dim, seq_len) -> (batch, d_model, n_patches)
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)  # (batch, n_patches, d_model)
        
        # Layer norm and positional encoding
        x = self.input_norm(x)
        x = self.pos_encoding(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Output norm
        x = self.output_norm(x)
        
        # Output based on horizon
        if horizon is None:
            return self.single_head(x)
        elif horizon in self.forecast_horizons:
            return self.heads[f'h_{horizon}'](x)
        else:
            raise ValueError(f"Horizon {horizon} not in {self.forecast_horizons}")
    
    def forward_all_horizons(self, x):
        """Get predictions for all horizons at once"""
        # Patch embedding
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        
        # Layer norm and positional encoding
        x = self.input_norm(x)
        x = self.pos_encoding(x)
        
        # Transformer encoder
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_norm(x)
        
        # All horizon predictions
        return {h: self.heads[f'h_{h}'](x) for h in self.forecast_horizons}


class ImprovedPatchTSTWrapper:
    """Wrapper untuk ImprovedPatchTST dengan training utilities"""
    
    def __init__(self, input_dim, device=None, progress_callback=None, 
                 forecast_horizons=[20, 30, 60, 100], **kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.forecast_horizons = forecast_horizons
        
        lr = kwargs.pop('lr', 1e-3)
        weight_decay = kwargs.pop('weight_decay', 1e-5)
        
        self.model = ImprovedPatchTST(
            input_dim, 
            forecast_horizons=forecast_horizons, 
            **kwargs
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        self.progress_callback = progress_callback
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, 
            verbose=1, early_stopping_patience=10, horizon=None):
        """
        Train model dengan early stopping dan learning rate scheduling.
        
        Args:
            horizon: If None, train for single-step. Otherwise train for specific horizon.
        """
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            if y_val.ndim == 1:
                y_val = y_val.unsqueeze(1)
        
        n = X_train.shape[0]
        early_stopping = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            perm = torch.randperm(n)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, n, batch_size):
                xb = X_train_shuffled[i:i+batch_size]
                yb = y_train_shuffled[i:i+batch_size]
                
                self.optimizer.zero_grad()
                out = self.model(xb, horizon=horizon)
                
                # Adjust target shape for multi-horizon
                if horizon is not None and yb.shape[1] != out.shape[1]:
                    yb = yb[:, :out.shape[1]]
                
                loss = self.criterion(out, yb)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            
            # Validation phase
            val_loss = avg_train_loss  # Default if no validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val, horizon=horizon)
                    if horizon is not None and y_val.shape[1] != val_out.shape[1]:
                        y_val_adj = y_val[:, :val_out.shape[1]]
                    else:
                        y_val_adj = y_val
                    val_loss = self.criterion(val_out, y_val_adj).item()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if early_stopping(val_loss, self.model):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                early_stopping.restore_weights(self.model)
                break
            
            # Progress callback
            if self.progress_callback:
                progress = int((epoch + 1) / epochs * 100)
                self.progress_callback(
                    progress, 
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            elif verbose and (epoch % 10 == 0 or epoch == epochs-1):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs}, Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.6f}")
    
    def predict(self, X, horizon=None, batch_size=1024):
        """Predict dengan batched inference"""
        self.model.eval()
        
        if len(X) <= batch_size:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(X_tensor, horizon=horizon)
            return out.cpu().numpy()
        
        # Batched prediction
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
    
    def predict_all_horizons(self, X, batch_size=1024):
        """Predict for all horizons at once"""
        self.model.eval()
        
        all_results = {h: [] for h in self.forecast_horizons}
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            X_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model.forward_all_horizons(X_tensor)
            
            for h in self.forecast_horizons:
                all_results[h].append(outputs[h].cpu().numpy())
        
        return {h: np.concatenate(all_results[h]) for h in self.forecast_horizons}
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'forecast_horizons': self.forecast_horizons
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)


# Backward compatibility - keep original PatchTST
class PatchTST(nn.Module):
    def __init__(self, input_dim, patch_len=16, stride=8, d_model=128, n_heads=4, n_layers=2, dropout=0.1, out_dim=1):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.input_dim = input_dim
        self.patch_embed = nn.Conv1d(input_dim, d_model, kernel_size=patch_len, stride=stride)
        
        # Add positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoding(x)  # Add positional encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.head(x)
        return out


class PatchTSTWrapper:
    """Original wrapper with early stopping added"""
    
    def __init__(self, input_dim, device=None, progress_callback=None, **kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        lr = kwargs.pop('lr', 1e-3)
        self.model = PatchTST(input_dim, **kwargs).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.progress_callback = progress_callback

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, 
            verbose=1, early_stopping_patience=10):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        
        # Validation data
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
            X_train = X_train[perm]
            y_train = y_train[perm]
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, n, batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validation and early stopping
            val_loss = avg_loss
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val)
                    val_loss = self.criterion(val_out, y_val).item()
            
            if early_stopping(val_loss, self.model):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                early_stopping.restore_weights(self.model)
                break
            
            if self.progress_callback:
                progress = int((epoch + 1) / epochs * 100)
                self.progress_callback(progress, f"Melatih model (Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f})")
            elif verbose and (epoch % 10 == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X, batch_size=1024):
        self.model.eval()
        
        if len(X) <= batch_size:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(X_tensor)
            return out.cpu().numpy().flatten()
        
        all_preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            X_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(X_tensor)
            all_preds.append(out.cpu().numpy().flatten())
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return np.concatenate(all_preds)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)


# Hyperparameter tuning interface
from itertools import product

def patchtst_hyperparameter_search(X_train, y_train, X_val, y_val, param_grid, max_trials=10, log_dir=None, progress_callback=None, default_params=None):
    tuning_method = param_grid.pop('tuning_method', 'grid') if 'tuning_method' in param_grid else 'grid'
    
    # Initialize with worst possible score
    best_score = float('inf')
    best_params = None
    best_model = None

    # 1. Evaluate Default Params First (Baseline)
    if default_params:
        if progress_callback:
            progress_callback(0, "Evaluating Default Parameters (Baseline)...")
        else:
            print(f"Evaluating Default Parameters: {default_params}")
            
        try:
            # Create and train default model
            baseline_model = PatchTSTWrapper(input_dim=X_train.shape[2], progress_callback=progress_callback, **default_params)
            baseline_model.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose=0)
            
            # Evaluate
            y_pred = baseline_model.predict(X_val)
            score = np.mean((y_pred - y_val.flatten())**2)
            
            if progress_callback:
                 progress_callback(5, f"Baseline MSE: {score:.4f}")
            else:
                 print(f"Baseline MSE: {score:.4f}")
                 
            best_score = score
            best_params = default_params
            best_model = baseline_model
        except Exception as e:
            print(f"Failed to evaluate baseline: {str(e)}")

    if tuning_method == 'bayesian':
        def objective(trial):
            params = {
                'patch_len': trial.suggest_int('patch_len', 8, 32),
                'stride': trial.suggest_int('stride', 2, 16),
                'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
                'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
                'n_layers': trial.suggest_int('n_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.05, 0.3),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            }
            model = PatchTSTWrapper(input_dim=X_train.shape[2], progress_callback=progress_callback, **params)
            model.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose=0)
            y_pred = model.predict(X_val)
            score = np.mean((y_pred - y_val.flatten())**2)
            return score
        
        study = optuna.create_study(direction='minimize')
        # Enqueue default params if available
        if default_params:
            study.enqueue_trial(default_params)
            
        study.optimize(objective, n_trials=max_trials)
        
        if study.best_value < best_score:
            best_params = study.best_trial.params
            best_model = PatchTSTWrapper(input_dim=X_train.shape[2], progress_callback=progress_callback, **best_params)
            best_model.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose=0)
            best_score = study.best_value
            
        return best_model, best_params, best_score
    else:
        # Grid Search
        keys, values = zip(*param_grid.items())
        trials = 0
        for v in product(*values):
            params = dict(zip(keys, v))
            if progress_callback:
                progress_callback(int(trials/max_trials * 100), f"Trial {trials+1}/{max_trials}: {params}")
            else:
                print(f"Trial {trials+1}/{max_trials}: {params}")
            model = PatchTSTWrapper(input_dim=X_train.shape[2], progress_callback=progress_callback, **params)
            model.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose=0)
            y_pred = model.predict(X_val)
            score = np.mean((y_pred - y_val.flatten())**2)
            if progress_callback:
                progress_callback(int((trials+1)/max_trials * 100), f"Trial {trials+1}/{max_trials} - MSE: {score:.4f}")
            else:
                print(f"Validation MSE: {score:.4f}")
            if score < best_score:
                best_score = score
                best_params = params
                best_model = model
            trials += 1
            if trials >= max_trials:
                break
        return best_model, best_params, best_score


# =============================================================================
# HELPER FUNCTIONS FOR FEATURE PREPARATION
# =============================================================================

def prepare_patchtst_input(prices, ohlcv_df=None, use_scores=True, sequence_length=60):
    """
    Prepare multivariate STATIONARY input for PatchTST model.
    
    This function creates feature-rich input sequences suitable for PatchTST.
    All features are designed to be stationary (log returns, z-scores, bounded scores).
    
    Args:
        prices: Array of Close prices
        ohlcv_df: Optional DataFrame with OHLCV data (more accurate indicators)
        use_scores: If True, use compact signal scores (8 features)
                   If False, use raw indicators (9 features)
        sequence_length: Length of input sequences
    
    Returns:
        X: np.ndarray of shape (n_samples, sequence_length, n_features)
        
    Features when use_scores=True (8 features, all bounded/stationary):
        - Log Returns
        - Rolling Z-Score (20-period)
        - MACD Score (0-1)
        - Stochastic RSI Score (0-1)
        - Bollinger Band Score (0-1)
        - Volume Score (0-1)
        - ADX Normalized (0-1)
        - BB %B (0-1)
    
    Example:
        >>> from src.models.patchtst_model import prepare_patchtst_input, ImprovedPatchTSTWrapper
        >>> X = prepare_patchtst_input(prices, sequence_length=60)
        >>> model = ImprovedPatchTSTWrapper(input_dim=X.shape[2])
        >>> model.fit(X_train, y_train, X_val, y_val)
    """
    if not HAS_FEATURE_ENGINEERING:
        raise ImportError(
            "Feature engineering module not available. "
            "Please ensure src/data/feature_engineering.py exists."
        )
    
    # Get features using TradingFeatureEngineer
    features = _prepare_features(prices, ohlcv_df=ohlcv_df, use_scores=use_scores)
    
    # Create sequences
    n_samples = len(features) - sequence_length
    n_features = features.shape[1]
    
    X = np.zeros((n_samples, sequence_length, n_features), dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = features[i:i+sequence_length]
    
    return X


def get_patchtst_feature_info(use_scores=True):
    """
    Get information about PatchTST input features.
    
    Returns:
        dict with feature names and descriptions
    """
    if use_scores:
        return {
            'n_features': 8,
            'features': [
                ('log_returns', 'Log returns (stationary)'),
                ('price_zscore', 'Rolling 20-period Z-score of price'),
                ('macd_score', 'MACD signal score (0-1, 0.5=neutral)'),
                ('stoch_rsi_score', 'Stochastic RSI score (0-1, 0.5=neutral)'),
                ('bb_score', 'Bollinger Band score (0-1, 0.5=neutral)'),
                ('volume_score', 'Volume confirmation score (0-1, 0.5=neutral)'),
                ('adx_normalized', 'ADX trend strength (0-1)'),
                ('bb_percent_b', 'BB %B price position (0-1)')
            ]
        }
    else:
        return {
            'n_features': 9,
            'features': [
                ('log_returns', 'Log returns (stationary)'),
                ('price_zscore', 'Rolling 20-period Z-score'),
                ('rsi_normalized', 'RSI / 100'),
                ('stoch_rsi_k', 'Stochastic RSI %K / 100'),
                ('stoch_rsi_d', 'Stochastic RSI %D / 100'),
                ('macd_histogram', 'Normalized MACD histogram'),
                ('adx_normalized', 'ADX / 100'),
                ('bb_percent_b', 'BB %B clipped to 0-1'),
                ('volume_ratio', 'Normalized volume ratio')
            ]
        }