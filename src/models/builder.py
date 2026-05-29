"""
Model Builder Module
==================

Modul ini berisi kelas ModelBuilder untuk membangun model PatchTST.
Model lain (CNN-LSTM, BiLSTM, Transformer, Ensemble) telah dihapus.
"""

from .patchtst_model import PatchTSTWrapper


class ModelBuilder:
    """Builder class untuk model prediksi harga saham."""
    
    @staticmethod
    def build_patchtst(input_shape, **kwargs):
        """
        Membangun model PatchTST.
        
        Parameters
        ----------
        input_shape : tuple
            Shape input data (sequence_length, num_features)
        **kwargs : dict
            Parameter tambahan untuk PatchTSTWrapper
            
        Returns
        -------
        PatchTSTWrapper
            Instance model PatchTST
        """
        # input_shape = (sequence_length, num_features)
        input_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        
        default_params = {
            'patch_len': 16,
            'stride': 8,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1,
            'lr': 1e-3,
        }
        
        # Override dengan kwargs
        default_params.update(kwargs)
        
        return PatchTSTWrapper(input_dim=input_dim, **default_params)
    
    # Aliases untuk kompatibilitas
    @staticmethod
    def build_default(input_shape, **kwargs):
        """Alias untuk build_patchtst (default model)."""
        return ModelBuilder.build_patchtst(input_shape, **kwargs)