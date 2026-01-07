"""
Models Module
============

Modul ini berisi kelas-kelas untuk membangun dan melatih
model prediksi harga saham.

Includes:
- StockPredictor: Main predictor class with data leakage fix
- PatchTSTWrapper: Original PatchTST wrapper
- ImprovedPatchTSTWrapper: Improved PatchTST with positional encoding
- EnsemblePredictor: Ensemble of PatchTST + BiLSTM + XGBoost
"""

from .predictor import StockPredictor
from .patchtst_model import PatchTSTWrapper, ImprovedPatchTSTWrapper, PatchTST, ImprovedPatchTST

try:
    from .ensemble_model import EnsemblePredictor, BiLSTMWrapper
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

__all__ = [
    'StockPredictor',
    'PatchTSTWrapper',
    'ImprovedPatchTSTWrapper',
    'PatchTST',
    'ImprovedPatchTST',
]

if ENSEMBLE_AVAILABLE:
    __all__.extend(['EnsemblePredictor', 'BiLSTMWrapper'])