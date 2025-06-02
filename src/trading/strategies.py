"""
Trading Strategies Module
=======================

Modul ini berisi implementasi berbagai strategi trading
yang digunakan dalam backtest.
"""

from .ppo_agent import PPOTrader
import numpy as np

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
    def ppo(predicted_prices, actual_prices, current_idx, params=None):
        """
        Strategi trading menggunakan PPO agent
        
        Parameters:
        -----------
        predicted_prices : array-like
            Array harga prediksi
        actual_prices : array-like
            Array harga aktual
        current_idx : int
            Indeks waktu saat ini
        params : dict, optional
            Parameter untuk PPO agent
            
        Returns:
        --------
        str
            Sinyal trading ('buy', 'sell', atau 'hold')
        """
        if params is None:
            params = {
                'training_done': False,
                'actor_lr': 0.0003,
                'critic_lr': 0.001,
                'gamma': 0.99,
                'clip_ratio': 0.2,
                'episodes': 10
            }
        
        # Jika training belum dilakukan, lakukan training
        if 'training_done' not in params or not params['training_done']:
            # Initialize PPO agent
            from .ppo_agent import PPOAgent
            agent = PPOAgent(
                state_dim=5,  # [price, predicted_price, return, volatility, position]
                action_dim=3,  # buy, sell, hold
                hidden_dim=64,
                lr=params['actor_lr'],
                gamma=params['gamma'],
                epsilon=params['clip_ratio']
            )
            
            # Prepare training data
            states = []
            for i in range(len(actual_prices)):
                if i > 0:
                    returns = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
                    volatility = np.std([
                        (actual_prices[j] - actual_prices[j-1]) / actual_prices[j-1]
                        for j in range(max(0, i-10), i)
                    ]) if i > 10 else 0
                else:
                    returns = 0
                    volatility = 0
                    
                state = np.array([
                    actual_prices[i],
                    predicted_prices[i],
                    returns,
                    volatility,
                    0  # initial position
                ])
                states.append(state)
                
            # Train agent
            agent.train(np.array(states), n_epochs=params['episodes'])
            params['agent'] = agent
            params['training_done'] = True
        
        # Get current state
        if current_idx > 0:
            returns = (actual_prices[current_idx] - actual_prices[current_idx-1]) / actual_prices[current_idx-1]
            volatility = np.std([
                (actual_prices[j] - actual_prices[j-1]) / actual_prices[j-1]
                for j in range(max(0, current_idx-10), current_idx)
            ]) if current_idx > 10 else 0
        else:
            returns = 0
            volatility = 0
            
        state = np.array([
            actual_prices[current_idx],
            predicted_prices[current_idx],
            returns,
            volatility,
            0  # current position
        ])
        
        # Get action from agent
        action, _ = params['agent'].get_action(state)
        
        return 'buy' if action == 0 else 'sell' if action == 1 else 'hold'
    
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