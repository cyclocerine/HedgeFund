"""
Trading Strategies Module
=======================

Modul ini berisi implementasi berbagai strategi trading
yang digunakan dalam backtest.
"""

from .ppo_agent import PPOTrader

class TradingStrategy:
    @staticmethod
    def trend_following(predicted_prices, actual_prices, index, params=None):
        """
        Strategi Trend Following
        
        Beli jika prediksi menunjukkan tren naik, jual jika prediksi menunjukkan tren turun.
        
        Parameters:
        -----------
        predicted_prices : array-like
            Harga prediksi dari model
        actual_prices : array-like
            Harga aktual historis
        index : int
            Indeks waktu saat ini
        params : dict, optional
            Parameter tambahan untuk strategi ini
            
        Returns:
        --------
        str
            Sinyal trading: 'BUY', 'SELL', atau 'HOLD'
        """
        if params is None:
            params = {'threshold': 0.01}
            
        threshold = params.get('threshold', 0.01)
        
        # Beli jika prediksi menunjukkan tren naik
        if index > 1 and predicted_prices[index] > predicted_prices[index-1] * (1 + threshold):
            return 'BUY'
        # Jual jika prediksi menunjukkan tren turun
        elif index > 1 and predicted_prices[index] < predicted_prices[index-1] * (1 - threshold):
            return 'SELL'
        
        return 'HOLD'
    
    @staticmethod
    def mean_reversion(predicted_prices, actual_prices, index, params=None):
        """
        Strategi Mean Reversion
        
        Beli jika harga di bawah rata-rata (oversold), jual jika harga di atas rata-rata (overbought).
        
        Parameters:
        -----------
        predicted_prices : array-like
            Harga prediksi dari model
        actual_prices : array-like
            Harga aktual historis
        index : int
            Indeks waktu saat ini
        params : dict, optional
            Parameter tambahan untuk strategi ini
            
        Returns:
        --------
        str
            Sinyal trading: 'BUY', 'SELL', atau 'HOLD'
        """
        if params is None:
            params = {'window': 5, 'buy_threshold': 0.98, 'sell_threshold': 1.02}
            
        window = params.get('window', 5)
        buy_threshold = params.get('buy_threshold', 0.98)
        sell_threshold = params.get('sell_threshold', 1.02)
        
        # Hitung rata-rata bergerak
        if index >= window:
            sma = sum(actual_prices[index-window:index]) / window
            # Beli jika harga di bawah rata-rata (oversold)
            if actual_prices[index] < sma * buy_threshold:
                return 'BUY'
            # Jual jika harga di atas rata-rata (overbought)
            elif actual_prices[index] > sma * sell_threshold:
                return 'SELL'
        
        return 'HOLD'
    
    @staticmethod
    def predictive(predicted_prices, actual_prices, index, params=None):
        """
        Strategi Predictive
        
        Beli jika prediksi menunjukkan harga akan naik, jual jika prediksi menunjukkan harga akan turun.
        
        Parameters:
        -----------
        predicted_prices : array-like
            Harga prediksi dari model
        actual_prices : array-like
            Harga aktual historis
        index : int
            Indeks waktu saat ini
        params : dict, optional
            Parameter tambahan untuk strategi ini
            
        Returns:
        --------
        str
            Sinyal trading: 'BUY', 'SELL', atau 'HOLD'
        """
        if params is None:
            params = {'buy_threshold': 1.01, 'sell_threshold': 0.99}
            
        buy_threshold = params.get('buy_threshold', 1.01)
        sell_threshold = params.get('sell_threshold', 0.99)
        
        # Beli jika prediksi menunjukkan harga akan naik
        if predicted_prices[index] > actual_prices[index] * buy_threshold:
            return 'BUY'
        # Jual jika prediksi menunjukkan harga akan turun
        elif predicted_prices[index] < actual_prices[index] * sell_threshold:
            return 'SELL'
        
        return 'HOLD'
    
    @staticmethod
    def ppo(predicted_prices, actual_prices, index, params=None):
        """
        Strategi PPO (Proximal Policy Optimization)
        
        Menggunakan reinforcement learning untuk menentukan sinyal trading optimal.
        
        Parameters:
        -----------
        predicted_prices : array-like
            Harga prediksi dari model
        actual_prices : array-like
            Harga aktual historis
        index : int
            Indeks waktu saat ini
        params : dict, optional
            Parameter tambahan untuk strategi ini
            
        Returns:
        --------
        str
            Sinyal trading: 'BUY', 'SELL', atau 'HOLD'
        """
        if params is None or 'ppo_agent' not in params:
            # Gunakan data sebelumnya untuk training jika belum ada agent
            if 'training_done' not in params or not params['training_done']:
                # Dapatkan data untuk training (historis sampai indeks saat ini)
                train_prices = actual_prices[:index+1]
                
                # Buat dan latih model PPO jika belum dilatih
                if index > 30:  # Pastikan data cukup untuk training
                    # Siapkan fitur, termasuk prediksi
                    import numpy as np
                    
                    # Gunakan predicted_prices sebagai fitur tambahan jika tersedia
                    if predicted_prices is not None:
                        # Pastikan bentuk array konsisten dengan mengubah menjadi vector kolom
                        pred_reshaped = np.array(predicted_prices[:index+1]).reshape(-1, 1)
                        actual_reshaped = np.array(actual_prices[:index+1]).reshape(-1, 1)
                        
                        # Gabungkan feature dengan bentuk yang konsisten
                        train_features = np.hstack((pred_reshaped, actual_reshaped))
                    else:
                        train_features = None
                    
                    # Buat PPOTrader baru
                    ppo_trader = PPOTrader(
                        prices=train_prices,
                        features=train_features,
                        initial_investment=10000
                    )
                    
                    # Latih model (jumlah episode lebih sedikit untuk runtime lebih cepat)
                    train_results = ppo_trader.train(episodes=10)
                    
                    # Simpan agent di params untuk digunakan selanjutnya
                    if 'ppo_agent' not in params:
                        params['ppo_agent'] = ppo_trader
                        params['training_done'] = True
                        params['actions'] = []
                
                # Jika tidak cukup data atau training belum selesai, gunakan strategi default
                return TradingStrategy.predictive(predicted_prices, actual_prices, index)
            
        # Gunakan agent yang sudah dilatih untuk menghasilkan sinyal
        if 'ppo_agent' in params:
            ppo_agent = params['ppo_agent']
            
            # Prepare state untuk agent
            if 'actions' not in params:
                params['actions'] = []
                
            # Buat observasi dari data saat ini
            features = None
            if predicted_prices is not None:
                # Jika ada prediksi, tambahkan sebagai fitur
                import numpy as np
                # Reshape menjadi 2D array dengan bentuk konsisten
                features = np.array([predicted_prices[index]]).reshape(1, 1)
            
            # Gunakan dummy environment untuk mendapatkan aksi
            env = ppo_agent.env
            env.prices = actual_prices
            env.current_step = index
            
            # Dapatkan observasi
            if features is not None:
                if index < len(env.features):
                    # Pastikan bentuk dimensi sesuai
                    feature_width = env.features.shape[1]
                    if feature_width == features.shape[1]:
                        env.features[index] = features
                    else:
                        # Sesuaikan bentuk jika tidak sesuai
                        env.features[index] = np.zeros(feature_width)
                        env.features[index, 0] = features[0, 0]
                else:
                    # Handle jika index diluar range
                    dummy_features = np.zeros((1, env.features.shape[1]))
                    dummy_features[0, 0] = features[0, 0]
                    env.features = np.concatenate([env.features, dummy_features])
                    
            # Get observation
            observation = env._get_observation()
            
            # Get action from agent
            action, _, _ = ppo_agent.agent.get_action(observation)
            params['actions'].append(action)
            
            # Convert action (0=HOLD, 1=BUY, 2=SELL) to signal
            if action == 1:
                return 'BUY'
            elif action == 2:
                return 'SELL'
            else:
                return 'HOLD'
        
        # Fallback ke strategi predictive
        return TradingStrategy.predictive(predicted_prices, actual_prices, index)
    
    @staticmethod
    def get_strategy_function(strategy_name):
        """
        Mendapatkan fungsi strategi berdasarkan nama
        
        Parameters:
        -----------
        strategy_name : str
            Nama strategi
            
        Returns:
        --------
        function
            Fungsi strategi yang sesuai
            
        Raises:
        -------
        ValueError
            Jika strategi tidak ditemukan
        """
        strategies = {
            'Trend Following': TradingStrategy.trend_following,
            'Mean Reversion': TradingStrategy.mean_reversion,
            'Predictive': TradingStrategy.predictive,
            'PPO': TradingStrategy.ppo,
            'trend_following': TradingStrategy.trend_following,
            'mean_reversion': TradingStrategy.mean_reversion,
            'predictive': TradingStrategy.predictive,
            'ppo': TradingStrategy.ppo
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Strategi '{strategy_name}' tidak ditemukan")
            
        return strategies[strategy_name] 