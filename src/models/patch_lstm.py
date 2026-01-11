"""
P-LSTM (Patch-LSTM) Model
=========================

A hybrid architecture that combines the patching strategy from PatchTST
with LSTM for efficient long-sequence processing.

Key Benefits:
- Reduces LSTM unroll steps from L to L/P (where P is patch size)
- Mitigates vanishing gradient problem on long sequences
- Adds residual connections for gradient flow

Author: AI Hedge Fund V2.3
"""

import torch
import torch.nn as nn
import numpy as np


class PatchLSTM(nn.Module):
    """
    Patch-LSTM model for time series forecasting.
    
    Instead of processing one timestep at a time (which causes LSTM to forget),
    we process patches (e.g., 16 days at once) as single tokens.
    
    Args:
        input_dim: Number of features per timestep
        patch_len: Length of each patch (default: 16)
        d_model: Hidden dimension (default: 128)
        lstm_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        bidirectional: Use bidirectional LSTM (default: True)
        forecast_horizons: List of forecast horizons (default: [1, 7, 14, 30])
    """
    
    def __init__(self, input_dim, patch_len=16, d_model=128, 
                 lstm_layers=2, dropout=0.1, bidirectional=True,
                 forecast_horizons=[1, 7, 14, 30]):
        super().__init__()
        
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.forecast_horizons = forecast_horizons
        
        # Patch Embedding: Linear projection of flattened patch
        self.patch_embed = nn.Linear(patch_len * input_dim, d_model)
        
        # Layer Normalization after embedding
        self.embed_norm = nn.LayerNorm(d_model)
        
        # LSTM for processing sequence of patches
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension (2x for bidirectional)
        lstm_out_dim = d_model * 2 if bidirectional else d_model
        
        # Residual projection (if dimensions differ)
        self.residual_proj = nn.Linear(d_model, lstm_out_dim) if bidirectional else nn.Identity()
        
        # Layer norm after LSTM
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Multi-horizon output heads
        self.output_heads = nn.ModuleDict({
            f"horizon_{h}": nn.Sequential(
                nn.Linear(lstm_out_dim, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            ) for h in forecast_horizons
        })
        
        # Single-step prediction head (for backward compatibility)
        self.single_head = nn.Linear(lstm_out_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier for better training."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
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
        
        # Trim to complete patches
        x = x[:, :n_patches * self.patch_len, :]
        
        # Reshape to patches: (B, n_patches, patch_len * input_dim)
        x_patched = x.view(B, n_patches, self.patch_len * D)
        
        # Patch embedding: (B, n_patches, d_model)
        x_embed = self.patch_embed(x_patched)
        x_embed = self.embed_norm(x_embed)
        
        # Store for residual connection
        residual = self.residual_proj(x_embed.mean(dim=1))  # Global average pool
        
        # LSTM processing: (B, n_patches, lstm_out_dim)
        lstm_out, (h_n, c_n) = self.lstm(x_embed)
        
        # Use last output + residual connection
        out = lstm_out[:, -1, :] + residual
        out = self.lstm_norm(out)
        out = self.dropout(out)
        
        # Select output head based on horizon
        if horizon is not None and f"horizon_{horizon}" in self.output_heads:
            return self.output_heads[f"horizon_{horizon}"](out)
        else:
            return self.single_head(out)
    
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
        
        # Trim and patch
        x = x[:, :n_patches * self.patch_len, :]
        x_patched = x.view(B, n_patches, self.patch_len * D)
        
        # Embed and process
        x_embed = self.patch_embed(x_patched)
        x_embed = self.embed_norm(x_embed)
        residual = self.residual_proj(x_embed.mean(dim=1))
        
        lstm_out, _ = self.lstm(x_embed)
        out = lstm_out[:, -1, :] + residual
        out = self.lstm_norm(out)
        out = self.dropout(out)
        
        # Get all horizon predictions
        predictions = {}
        for h in self.forecast_horizons:
            predictions[h] = self.output_heads[f"horizon_{h}"](out)
        
        return predictions


class PatchLSTMWrapper:
    """
    Wrapper class with training utilities for PatchLSTM.
    
    Provides fit(), predict(), save(), and load() methods
    similar to ImprovedPatchTSTWrapper for drop-in replacement.
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
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = nn.MSELoss()
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50,
            batch_size=32, verbose=1, early_stopping_patience=10, horizon=None):
        """
        Train the P-LSTM model.
        
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
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Shuffle data
            perm = torch.randperm(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                self.optimizer.zero_grad()
                pred = self.model(batch_X, horizon=horizon)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / n_batches
            
            # Validation
            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val, horizon=horizon)
                    val_loss = self.criterion(val_pred, y_val).item()
                
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_loss:.6f}" if val_loss else ""
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}{val_str}")
        
        # Restore best weights
        if best_state is not None:
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
    print("Testing P-LSTM...")
    
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
    
    print("P-LSTM test passed!")
