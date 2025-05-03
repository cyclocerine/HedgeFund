"""
Strategy Optimizer Module
======================

Modul ini berisi implementasi kelas StrategyOptimizer untuk
mengoptimalkan parameter strategi trading.
"""

import itertools
import numpy as np
from .strategies import TradingStrategy
from .ppo_agent import PPOTrader

class StrategyOptimizer:
    def __init__(self, actual_prices, predicted_prices, initial_investment=10000):
        """
        Inisialisasi StrategyOptimizer
        
        Parameters:
        -----------
        actual_prices : array-like
            Array harga aktual historis
        predicted_prices : array-like
            Array harga prediksi dari model
        initial_investment : float, optional
            Jumlah investasi awal, default 10000
        """
        self.actual_prices = actual_prices
        self.predicted_prices = predicted_prices
        self.initial_investment = initial_investment
        
    def optimize(self, strategy, param_ranges):
        """
        Mencari parameter optimal untuk strategi
        
        Parameters:
        -----------
        strategy : str
            Nama strategi yang akan dioptimalkan
        param_ranges : dict
            Dictionary berisi range parameter yang akan diuji
            
        Returns:
        --------
        tuple
            (best_params, best_performance)
            - best_params: parameter terbaik untuk strategi
            - best_performance: metrik performa terbaik
        """
        # Validasi input
        if not param_ranges:
            raise ValueError("Parameter ranges must be provided")
            
        # Khusus untuk strategi PPO perlu pendekatan berbeda
        if strategy.lower() == 'ppo':
            return self.optimize_ppo(param_ranges)
            
        # Siapkan kombinasi parameter
        param_combinations = self._generate_parameter_combinations(param_ranges)
        
        # Dapatkan fungsi strategi
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        
        # Uji setiap kombinasi parameter
        best_return = -np.inf
        best_params = None
        best_performance = None
        best_portfolio_values = None
        best_trades = None
        
        for params in param_combinations:
            portfolio_values, trades, performance = self.run_backtest(
                strategy_function, params
            )
            
            # Cari return terbaik
            current_return = performance['total_return']
            if current_return > best_return:
                best_return = current_return
                best_params = params
                best_performance = performance
                best_portfolio_values = portfolio_values
                best_trades = trades
        
        return best_params, best_performance, best_portfolio_values, best_trades
        
    def optimize_ppo(self, param_ranges):
        """
        Mengoptimalkan parameter untuk strategi PPO
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary berisi range parameter yang akan diuji
            
        Returns:
        --------
        tuple
            (best_params, best_performance, best_portfolio_values, best_trades)
        """
        # Extract ppo specific params
        ppo_params = {
            'actor_lr': param_ranges.get('actor_lr', [0.0003]),
            'critic_lr': param_ranges.get('critic_lr', [0.001]),
            'gamma': param_ranges.get('gamma', [0.99]),
            'clip_ratio': param_ranges.get('clip_ratio', [0.2]),
            'episodes': param_ranges.get('episodes', [50]),
        }
        
        # Generate parameter combinations
        ppo_param_combinations = self._generate_parameter_combinations(ppo_params)
        
        best_return = -np.inf
        best_params = None
        best_performance = None
        best_portfolio_values = None
        best_trades = None
        
        # Create features for PPO
        train_size = int(len(self.actual_prices) * 0.8)
        
        # Split data for training and evaluation
        train_prices = self.actual_prices[:train_size]
        train_predicted = self.predicted_prices[:train_size] if self.predicted_prices is not None else None
        
        eval_prices = self.actual_prices[train_size:]
        eval_predicted = self.predicted_prices[train_size:] if self.predicted_prices is not None else None
        
        # Persiapan fitur
        import numpy as np
        train_features = None
        if train_predicted is not None:
            train_features = np.column_stack((
                train_predicted,  # Prediksi harga
                train_prices      # Harga aktual
            ))
            
        # Uji setiap kombinasi parameter
        for params in ppo_param_combinations:
            try:
                # Buat dan latih model PPO
                ppo_trader = PPOTrader(
                    prices=train_prices,
                    features=train_features,
                    initial_investment=self.initial_investment
                )
                
                # Set parameter agent
                ppo_trader.agent.actor_lr = params['actor_lr']
                ppo_trader.agent.critic_lr = params['critic_lr']
                ppo_trader.agent.gamma = params['gamma']
                ppo_trader.agent.clip_ratio = params['clip_ratio']
                
                # Latih model
                train_results = ppo_trader.train(episodes=params['episodes'])
                
                # Backtest model pada data evaluasi
                eval_features = None
                if eval_predicted is not None:
                    eval_features = np.column_stack((
                        eval_predicted,  # Prediksi harga
                        eval_prices      # Harga aktual
                    ))
                
                backtest_results = ppo_trader.backtest(
                    prices=eval_prices,
                    features=eval_features,
                    initial_investment=self.initial_investment
                )
                
                # Ambil hasil
                portfolio_values = backtest_results['portfolio_values']
                trades = backtest_results['trades']
                performance = backtest_results['performance']
                
                # Cari return terbaik
                current_return = performance['total_return']
                if current_return > best_return:
                    best_return = current_return
                    best_params = params
                    best_performance = performance
                    best_portfolio_values = portfolio_values
                    best_trades = trades
                    
                    # Tambahkan model terbaik ke params
                    best_params['ppo_agent'] = ppo_trader
                    
            except Exception as e:
                print(f"Error during PPO optimization: {str(e)}")
                continue
        
        return best_params, best_performance, best_portfolio_values, best_trades
    
    def _generate_parameter_combinations(self, param_ranges):
        """
        Menghasilkan semua kombinasi parameter dari range yang diberikan
        
        Parameters:
        -----------
        param_ranges : dict
            Dictionary berisi range parameter yang akan diuji
            
        Returns:
        --------
        list
            List kombinasi parameter
        """
        # Ekstrak nama parameter dan nilai-nilainya
        param_names = []
        param_values = []
        
        for name, values in param_ranges.items():
            param_names.append(name)
            param_values.append(values)
        
        # Buat semua kombinasi
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
        return combinations
    
    def run_backtest(self, strategy_function, params):
        """
        Menjalankan backtest dengan parameter tertentu
        
        Parameters:
        -----------
        strategy_function : function
            Fungsi strategi yang akan digunakan
        params : dict
            Parameter untuk strategi
            
        Returns:
        --------
        tuple
            (portfolio_values, trades, performance)
            - portfolio_values: nilai portfolio per hari
            - trades: list transaksi yang dilakukan
            - performance: metrik performa
        """
        # Inisialisasi portfolio
        cash = self.initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        # Memastikan data memiliki panjang yang sama
        length = min(len(self.actual_prices), len(self.predicted_prices))
        actual_prices = self.actual_prices[:length]
        predicted_prices = self.predicted_prices[:length]
        
        # Iterasi melalui harga historis
        for i in range(1, length):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * actual_prices[i]
            portfolio_values.append(portfolio_value)
            
            # Generate signal
            signal = strategy_function(predicted_prices, actual_prices, i, params)
            
            # Proses signals
            if signal == 'BUY' and cash > 0:
                # Beli saham sebanyak mungkin dengan uang yang ada
                shares_to_buy = cash / actual_prices[i]
                shares += shares_to_buy
                cash = 0
                trades.append({
                    'day': i,
                    'type': 'BUY',
                    'price': actual_prices[i],
                    'shares': shares_to_buy,
                    'value': shares_to_buy * actual_prices[i]
                })
            elif signal == 'SELL' and shares > 0:
                # Jual semua saham
                cash += shares * actual_prices[i]
                trades.append({
                    'day': i,
                    'type': 'SELL',
                    'price': actual_prices[i],
                    'shares': shares,
                    'value': shares * actual_prices[i]
                })
                shares = 0
        
        # Nilai akhir portfolio
        final_value = cash + shares * actual_prices[-1]
        
        # Menghitung metrik performa
        performance = self._calculate_performance(portfolio_values, trades, final_value)
        
        return portfolio_values, trades, performance
    
    def _calculate_performance(self, portfolio_values, trades, final_value):
        """
        Menghitung metrik performa
        
        Parameters:
        -----------
        portfolio_values : array-like
            Nilai portfolio per hari
        trades : list
            List transaksi yang dilakukan
        final_value : float
            Nilai akhir portfolio
            
        Returns:
        --------
        dict
            Dictionary berisi metrik performa
        """
        # Menghitung return total
        total_return = (final_value - self.initial_investment) / self.initial_investment * 100
        
        # Menghitung drawdown
        peak = portfolio_values[0] if portfolio_values else self.initial_investment
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
        # Menghitung Sharpe Ratio
        if len(portfolio_values) > 1:
            daily_returns = [portfolio_values[i]/portfolio_values[i-1]-1 for i in range(1, len(portfolio_values))]
            if daily_returns and np.std(daily_returns) > 0:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Menghitung profit/loss per trade
        win_trades = 0
        loss_trades = 0
        for i in range(0, len(trades), 2):
            if i+1 < len(trades):
                buy = trades[i]
                sell = trades[i+1]
                profit = sell['value'] - buy['value']
                if profit > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
        
        win_rate = 0
        if win_trades + loss_trades > 0:
            win_rate = win_trades / (win_trades + loss_trades) * 100
        
        performance = {
            'initial_investment': self.initial_investment,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return performance 