import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from torch.utils.tensorboard import SummaryWriter

class PatchTST(nn.Module):
    def __init__(self, input_dim, patch_len=16, stride=8, d_model=128, n_heads=4, n_layers=2, dropout=0.1, out_dim=1):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.input_dim = input_dim
        self.patch_embed = nn.Conv1d(input_dim, d_model, kernel_size=patch_len, stride=stride)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.patch_embed(x)  # (batch, d_model, n_patches)
        x = x.permute(0, 2, 1)  # (batch, n_patches, d_model)
        x = self.transformer(x)  # (batch, n_patches, d_model)
        x = x.mean(dim=1)        # (batch, d_model)
        out = self.head(x)       # (batch, out_dim)
        return out

class PatchTSTWrapper:
    def __init__(self, input_dim, device=None, **kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        lr = kwargs.pop('lr', 1e-3)  # Ambil dan hapus 'lr' dari kwargs
        self.model = PatchTST(input_dim, **kwargs).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=1):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        n = X_train.shape[0]
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            X_train = X_train[perm]
            y_train = y_train[perm]
            for i in range(0, n, batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
            if verbose and (epoch % 10 == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(X)
        return out.cpu().numpy().flatten()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

# Hyperparameter tuning interface
from itertools import product

def patchtst_hyperparameter_search(X_train, y_train, X_val, y_val, param_grid, max_trials=10, log_dir=None):
    tuning_method = param_grid.pop('tuning_method', 'grid') if 'tuning_method' in param_grid else 'grid'
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
            model = PatchTSTWrapper(input_dim=X_train.shape[2], **params)
            writer = SummaryWriter(log_dir) if log_dir else None
            X_torch = torch.tensor(X_train, dtype=torch.float32).to(model.device)
            y_torch = torch.tensor(y_train, dtype=torch.float32).to(model.device)
            if y_torch.ndim == 1:
                y_torch = y_torch.unsqueeze(1)
            n = X_torch.shape[0]
            for epoch in range(20):
                model.model.train()
                perm = torch.randperm(n)
                X_torch = X_torch[perm]
                y_torch = y_torch[perm]
                for i in range(0, n, 32):
                    xb = X_torch[i:i+32]
                    yb = y_torch[i:i+32]
                    model.optimizer.zero_grad()
                    out = model.model(xb)
                    loss = model.criterion(out, yb)
                    loss.backward()
                    model.optimizer.step()
                if writer:
                    writer.add_scalar('Loss/train', loss.item(), epoch)
            if writer:
                writer.close()
            y_pred = model.predict(X_val)
            score = np.mean((y_pred - y_val.flatten())**2)
            return score
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=max_trials)
        best_params = study.best_trial.params
        best_model = PatchTSTWrapper(input_dim=X_train.shape[2], **best_params)
        best_model.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose=0)
        best_score = study.best_value
        return best_model, best_params, best_score
    else:
        best_score = float('inf')
        best_params = None
        best_model = None
        keys, values = zip(*param_grid.items())
        trials = 0
        for v in product(*values):
            params = dict(zip(keys, v))
            print(f"Trial {trials+1}/{max_trials}: {params}")
            model = PatchTSTWrapper(input_dim=X_train.shape[2], **params)
            model.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=32, verbose=0)
            y_pred = model.predict(X_val)
            score = np.mean((y_pred - y_val.flatten())**2)
            print(f"Validation MSE: {score:.4f}")
            if score < best_score:
                best_score = score
                best_params = params
                best_model = model
            trials += 1
            if trials >= max_trials:
                break
        return best_model, best_params, best_score 