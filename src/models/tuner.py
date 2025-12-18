"""
Hyperparameter Tuner Module
=========================

Modul ini berisi fungsi tuning untuk model PatchTST.
Tuning untuk model lain (CNN-LSTM, BiLSTM, Transformer) telah dihapus.
"""

from .patchtst_model import patchtst_hyperparameter_search, PatchTSTWrapper


class HyperparameterTuner:
    """
    Hyperparameter tuner untuk model PatchTST.
    """
    def __init__(self, input_shape, tuning_method='bayesian'):
        """
        Initialize tuner.
        
        Parameters
        ----------
        input_shape : tuple
            Shape input data (sequence_length, num_features)
        tuning_method : str
            Metode tuning: 'bayesian' atau 'grid'
        """
        self.input_shape = input_shape
        self.tuning_method = tuning_method
        self.input_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]

    def tune_patchtst(self, X_train, y_train, X_val, y_val, param_grid=None, max_trials=20):
        """
        Tuning hyperparameter untuk model PatchTST.
        
        Parameters
        ----------
        X_train : np.ndarray
            Data training
        y_train : np.ndarray
            Label training
        X_val : np.ndarray
            Data validasi
        y_val : np.ndarray
            Label validasi
        param_grid : dict, optional
            Grid parameter untuk tuning
        max_trials : int
            Jumlah maksimum trial
            
        Returns
        -------
        tuple
            (best_model, best_params, best_score)
        """
        if param_grid is None:
            param_grid = {
                'patch_len': [8, 16, 32],
                'stride': [4, 8, 16],
                'd_model': [64, 128, 256],
                'n_heads': [2, 4, 8],
                'n_layers': [2, 3, 4],
                'd_ff': [128, 256, 512],
                'dropout': [0.1, 0.2, 0.3],
                'lr': [1e-4, 5e-4, 1e-3],
                'tuning_method': self.tuning_method
            }
        else:
            param_grid['tuning_method'] = self.tuning_method
        
        return patchtst_hyperparameter_search(
            X_train, y_train, X_val, y_val, 
            param_grid, 
            max_trials=max_trials
        )

    def tune_model(self, model_type, X_train, y_train, X_val, y_val, max_trials=20, **kwargs):
        """
        Tune model berdasarkan tipe.
        
        Parameters
        ----------
        model_type : str
            Tipe model ('patchtst')
        X_train, y_train : np.ndarray
            Data training
        X_val, y_val : np.ndarray
            Data validasi
        max_trials : int
            Jumlah maksimum trial
            
        Returns
        -------
        Model yang sudah di-tune
        """
        if model_type != 'patchtst':
            print(f"[!] Model type '{model_type}' tidak tersedia. Menggunakan 'patchtst'.")
            model_type = 'patchtst'
        
        model, params, score = self.tune_patchtst(
            X_train, y_train, X_val, y_val, 
            param_grid=kwargs.get('param_grid'),
            max_trials=max_trials
        )
        
        return model