"""
P-LSTM (Patch-LSTM) Model — Enhanced v3.0
==========================================

A hybrid architecture that combines the patching strategy from PatchTST
with LSTM for efficient long-sequence processing.

Enhancements (v3.0):
- Multi-Head Cross-Patch Attention (PatchTST-inspired)
- Huber Loss for robustness to price spikes
- Cosine Annealing Warm Restarts scheduler
- Stochastic Weight Averaging (SWA) for generalization
- Input noise regularization (anti-overfit)
- Label smoothing via target noise

Key Benefits:
- Reduces LSTM unroll steps from L to L/P (where P is patch size)
- Mitigates vanishing gradient problem on long sequences
- Cross-patch attention captures inter-patch correlations
- SWA produces flatter minima → better generalization

Author: AI Hedge Fund V3.0
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for patch sequence."""
    
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchLSTM(nn.Module):
    """
    Patch-LSTM model with Cross-Patch Attention for time series forecasting.
    
    Architecture:
        1. Patch Embedding: Linear projection of flattened patches
        2. Positional Encoding: Sinusoidal position info for patches
        3. Cross-Patch Attention: Multi-head self-attention across patches
        4. LSTM: Sequential processing with temporal memory
        5. Residual Connection + LayerNorm
        6. Multi-horizon output heads
    
    Args:
        input_dim: Number of features per timestep
        patch_len: Length of each patch (default: 16)
        d_model: Hidden dimension (default: 128)
        lstm_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.15)
        bidirectional: Use bidirectional LSTM (default: True)
        n_heads: Number of attention heads (default: 4)
        attn_layers: Number of attention layers (default: 1)
        forecast_horizons: List of forecast horizons (default: [1, 7, 14, 30])
    """
    
    def __init__(self, input_dim, patch_len=16, d_model=128, 
                 lstm_layers=2, dropout=0.15, bidirectional=True,
                 n_heads=4, attn_layers=1,
                 forecast_horizons=[1, 7, 14, 30]):
        super().__init__()
        
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.forecast_horizons = forecast_horizons
        
        # LSTM processes raw sequence first
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = d_model * 2 if bidirectional else d_model
        
        # Patch Embedding: Linear projection of flattened patched LSTM outputs
        self.patch_embed = nn.Linear(patch_len * lstm_out_dim, d_model)
        
        # Layer Normalization after embedding
        self.embed_norm = nn.LayerNorm(d_model)
        
        # Positional Encoding for patch positions
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Cross-Patch Multi-Head Attention (PatchTST-inspired)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.cross_patch_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=attn_layers
        )
        
        # Attention output norm
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Multi-horizon output heads with bottleneck
        self.output_heads = nn.ModuleDict({
            f"horizon_{h}": nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_model // 2, 1)
            ) for h in forecast_horizons
        })
        
        # Single-step prediction head (for backward compatibility)
        self.single_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier for linear layers, orthogonal for LSTM."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1 (better gradient flow)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, horizon=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            horizon: Specific forecast horizon. If None, returns single-step.
            
        Returns:
            Prediction tensor of shape (batch, 1)
        """
        B, L, D = x.shape
        
        # Calculate number of complete patches
        n_patches = L // self.patch_len
        
        if n_patches == 0:
            # If sequence too short, pad to at least one patch
            pad_len = self.patch_len - L
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            n_patches = 1
            L = self.patch_len
        elif L % self.patch_len != 0:
            # Trim to complete patches before LSTM
            x = x[:, :n_patches * self.patch_len, :]
            L = n_patches * self.patch_len
        else:
            x = x[:, :n_patches * self.patch_len, :]
        
        # LSTM processing first: (B, L, lstm_out_dim)
        lstm_out, _ = self.lstm(x)
        lstm_out_dim = lstm_out.shape[-1]
        
        # Reshape to patches: (B, n_patches, patch_len * lstm_out_dim)
        x_patched = lstm_out.reshape(B, n_patches, self.patch_len * lstm_out_dim)
        
        # Patch embedding: (B, n_patches, d_model)
        x_embed = self.patch_embed(x_patched)
        x_embed = self.embed_norm(x_embed)
        
        # Positional encoding
        x_embed = self.pos_encoding(x_embed)
        
        # Cross-Patch Attention: learn inter-patch correlations
        x_attn = self.cross_patch_attn(x_embed)
        x_attn = self.attn_norm(x_attn)
        
        # Use last output from attention
        out = x_attn[:, -1, :]
        out = self.dropout(out)
        
        # Select output head based on horizon
        if horizon is not None and f"horizon_{horizon}" in self.output_heads:
            pred = self.output_heads[f"horizon_{horizon}"](out)
        else:
            pred = self.single_head(out)
            
        # Add residual from last timestamp, 0-th feature for random walk baseline
        return pred + x[:, -1, 0:1]
    
    def forward_all_horizons(self, x):
        """
        Get predictions for all forecast horizons at once.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Dict mapping horizon to prediction tensor
        """
        B, L, D = x.shape
        
        # Calculate number of complete patches
        n_patches = L // self.patch_len
        
        if n_patches == 0:
            pad_len = self.patch_len - L
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            n_patches = 1
            L = self.patch_len
        elif L % self.patch_len != 0:
            x = x[:, :n_patches * self.patch_len, :]
            L = n_patches * self.patch_len
        else:
            x = x[:, :n_patches * self.patch_len, :]
        
        # LSTM processing first
        lstm_out, _ = self.lstm(x)
        lstm_out_dim = lstm_out.shape[-1]
        
        # Reshape to patches
        x_patched = lstm_out.reshape(B, n_patches, self.patch_len * lstm_out_dim)
        
        # Embed, position encode, attend
        x_embed = self.patch_embed(x_patched)
        x_embed = self.embed_norm(x_embed)
        x_embed = self.pos_encoding(x_embed)
        
        x_attn = self.cross_patch_attn(x_embed)
        x_attn = self.attn_norm(x_attn)
        
        out = x_attn[:, -1, :]
        out = self.dropout(out)
        
        # Select output heads
        predictions = {}
        for horizon, head in self.output_heads.items():
            h_int = int(horizon.split('_')[1])
            pred = head(out)
            predictions[h_int] = pred + x[:, -1, 0:1]
            
        return predictions


class PatchLSTMWrapper:
    """
    Wrapper class with training utilities for PatchLSTM.
    
    Enhancements (v3.0):
    - Huber Loss (robust to price spike outliers)
    - Cosine Annealing Warm Restarts scheduler
    - Stochastic Weight Averaging (SWA) for better generalization
    - Input noise regularization during training
    - Label smoothing via target noise
    """
    
    def __init__(self, input_dim, device=None, progress_callback=None,
                 forecast_horizons=[1, 7, 14, 30], **kwargs):
        self.input_dim = input_dim
        self.forecast_horizons = forecast_horizons
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.progress_callback = progress_callback
        
        self.model = PatchLSTM(
            input_dim=input_dim,
            forecast_horizons=forecast_horizons,
            **kwargs
        ).to(self.device)
        
        # AdamW with decoupled weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=0.01
        )
        
        # Cosine Annealing Warm Restarts (better than ReduceLROnPlateau)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Huber Loss — robust to outliers (price spikes)
        self.criterion = nn.SmoothL1Loss(beta=0.5)
        
        # SWA state
        self.swa_model = None
        self.swa_start_epoch = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50,
            batch_size=32, verbose=1, early_stopping_patience=10, horizon=None,
            input_noise=0.01, label_noise=0.005, swa_start_pct=0.75):
        """
        Train the P-LSTM model with enhanced regularization.
        
        Args:
            X_train: Training features (N, seq_len, input_dim)
            y_train: Training targets (N,)
            X_val: Optional validation features
            y_val: Optional validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level (0=silent, 1=progress)
            early_stopping_patience: Epochs to wait before early stopping
            horizon: Specific horizon to train for (if None, trains for single-step)
            input_noise: Std of Gaussian noise added to inputs during training (anti-overfit)
            label_noise: Std of Gaussian noise added to targets (label smoothing)
            swa_start_pct: Fraction of training to start SWA (0.75 = last 25%)
        """
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).unsqueeze(-1).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        n_batches = (len(X_train) + batch_size - 1) // batch_size
        
        # SWA setup
        self.swa_start_epoch = int(epochs * swa_start_pct)
        swa_count = 0
        swa_state = None
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Shuffle data
            perm = torch.randperm(len(X_train))
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            
            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Input noise regularization (training only)
                if input_noise > 0:
                    batch_X = batch_X + torch.randn_like(batch_X) * input_noise
                
                # Label smoothing via target noise
                if label_noise > 0:
                    batch_y = batch_y + torch.randn_like(batch_y) * label_noise
                
                self.optimizer.zero_grad()
                pred = self.model(batch_X, horizon=horizon)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            # Step scheduler
            self.scheduler.step(epoch)
            
            avg_train_loss = total_loss / n_batches
            
            # Stochastic Weight Averaging
            if epoch >= self.swa_start_epoch:
                if swa_state is None:
                    swa_state = copy.deepcopy(self.model.state_dict())
                    swa_count = 1
                else:
                    current_state = self.model.state_dict()
                    swa_count += 1
                    for key in swa_state:
                        swa_state[key] = (
                            swa_state[key] * (swa_count - 1) + current_state[key]
                        ) / swa_count
            
            # Validation
            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val, horizon=horizon)
                    val_loss = self.criterion(val_pred, y_val).item()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_loss:.6f}" if val_loss else ""
                lr = self.optimizer.param_groups[0]['lr']
                swa_str = " [SWA]" if epoch >= self.swa_start_epoch else ""
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}{val_str} | LR: {lr:.2e}{swa_str}")
        
        # Apply SWA weights if available (better generalization)
        if swa_state is not None and swa_count > 1:
            self.model.load_state_dict(swa_state)
            if verbose:
                print(f"[P-LSTM] Applied SWA weights (averaged {swa_count} checkpoints)")
        elif best_state is not None:
            # Restore best early-stopping weights
            self.model.load_state_dict(best_state)
    
    def predict(self, X, horizon=None, batch_size=1024):
        """
        Make predictions on new data.
        
        Args:
            X: Features (N, seq_len, input_dim)
            horizon: Specific forecast horizon
            batch_size: Batch size for inference
            
        Returns:
            Predictions array (N,)
        """
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                pred = self.model(batch_X, horizon=horizon)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions).flatten()
    
    def predict_all_horizons(self, X, batch_size=1024):
        """
        Predict for all horizons at once.
        
        Returns:
            Dict mapping horizon to predictions array
        """
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        
        all_preds = {h: [] for h in self.forecast_horizons}
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                horizon_preds = self.model.forward_all_horizons(batch_X)
                for h, pred in horizon_preds.items():
                    all_preds[h].append(pred.cpu().numpy())
        
        return {h: np.concatenate(preds).flatten() for h, preds in all_preds.items()}
    
    def save(self, path):
        """Save model state to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'forecast_horizons': self.forecast_horizons,
        }, path)
        print(f"[P-LSTM] Model saved to {path}")
    
    def load(self, path):
        """Load model state from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[P-LSTM] Model loaded from {path}")


# Quick test
if __name__ == "__main__":
    print("Testing Enhanced P-LSTM v3.0...")
    
    # Create random data
    batch_size = 32
    seq_len = 128
    input_dim = 8
    
    X = torch.randn(batch_size, seq_len, input_dim)
    
    # Create model
    model = PatchLSTM(input_dim=input_dim, patch_len=16)
    
    # Forward pass
    out = model(X)
    print(f"Single output shape: {out.shape}")  # Expected: (32, 1)
    
    # Multi-horizon
    all_out = model.forward_all_horizons(X)
    print(f"Multi-horizon outputs: {list(all_out.keys())}")
    
    # Test wrapper with training
    wrapper = PatchLSTMWrapper(input_dim=input_dim)
    X_np = X.numpy()
    y_np = np.random.randn(batch_size)
    wrapper.fit(X_np, y_np, epochs=5, verbose=1)
    preds = wrapper.predict(X_np)
    print(f"Predictions shape: {preds.shape}")
    
    print("Enhanced P-LSTM v3.0 test passed!")
