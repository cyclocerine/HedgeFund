"""
Worker Threads Module
====================

Modul ini berisi implementasi thread worker untuk operasi asinkron di UI aplikasi.
"""

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class WorkerThread(QThread):
    """Thread worker untuk menjalankan prediksi saham"""
    finished = pyqtSignal(object, object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, predictor):
        """
        Inisialisasi worker thread
        
        Parameters:
        -----------
        predictor : StockPredictor
            Objek predictor yang akan digunakan
        """
        super().__init__()
        self.predictor = predictor
        
    def run(self):
        """Jalankan proses prediksi"""
        try:
            # Prepare data
            self.progress.emit(10)
            if not self.predictor.prepare_data():
                self.error.emit("Error preparing data")
                return
                
            # Train model
            self.progress.emit(30)
            history = self.predictor.train_model()
            self.progress.emit(70)
            
            # Make predictions
            y_true, y_pred, forecast = self.predictor.predict()
            self.progress.emit(90)
            
            # Calculate metrics
            metrics = self.predictor.evaluate(y_true, y_pred)
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(y_true, y_pred, forecast, metrics)
        except Exception as e:
            self.error.emit(str(e))

class BacktestWorker(QThread):
    """Thread worker untuk menjalankan backtesting"""
    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, predictor, initial_investment, strategy):
        """
        Inisialisasi worker thread
        
        Parameters:
        -----------
        predictor : StockPredictor
            Objek predictor yang berisi data
        initial_investment : float
            Jumlah investasi awal
        strategy : str
            Nama strategi trading yang akan digunakan
        """
        super().__init__()
        self.predictor = predictor
        self.initial_investment = initial_investment
        self.strategy = strategy
        
    def run(self):
        """Jalankan proses backtesting"""
        try:
            # Extract required data
            self.progress.emit(10)
            actual_prices = self.predictor.scaler.inverse_transform(
                np.concatenate([
                    self.predictor.y.reshape(-1, 1),
                    np.zeros((len(self.predictor.y), self.predictor.X.shape[2]-1))
                ], axis=1)
            )[:, 0]
            
            y_pred = self.predictor.model.predict(self.predictor.X)
            predicted_prices = self.predictor.scaler.inverse_transform(
                np.concatenate([
                    y_pred,
                    np.zeros((len(y_pred), self.predictor.X.shape[2]-1))
                ], axis=1)
            )[:, 0]
            
            self.progress.emit(30)
            
            # Run backtest
            portfolio_values, trades, performance = self.run_backtest(
                actual_prices, predicted_prices, self.initial_investment, self.strategy
            )
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(portfolio_values, trades, performance)
        except Exception as e:
            self.error.emit(str(e))
    
    def run_backtest(self, actual_prices, predicted_prices, initial_investment, strategy):
        """Implementasi backtest"""
        # Inisialisasi portfolio
        cash = initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        # Iterasi melalui harga historis
        for i in range(1, len(actual_prices)):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * actual_prices[i]
            portfolio_values.append(portfolio_value)
            
            # Generate signal
            signal = self.generate_signal(predicted_prices, actual_prices, i, strategy)
            
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
        # Menghitung return total
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Menghitung drawdown
        peak = portfolio_values[0] if portfolio_values else initial_investment
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
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
            'initial_investment': initial_investment,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance

    def generate_signal(self, predicted_prices, actual_prices, index, strategy):
        """Generate signal berdasarkan strategi"""
        from src.trading.strategies import TradingStrategy
        
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        # Inisialisasi params untuk menghindari NoneType error
        params = {}
        if strategy.lower() == 'ppo':
            params = {'training_done': False}
        return strategy_function(predicted_prices, actual_prices, index, params)

class StrategyOptimizerWorker(QThread):
    """Thread worker untuk menjalankan optimasi strategi"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, ticker, start_date, end_date, model, initial_investment, strategy=None,
                param_ranges=None):
        """
        Inisialisasi worker thread
        
        Parameters:
        -----------
        ticker : str
            Ticker saham
        start_date : str
            Tanggal mulai
        end_date : str
            Tanggal akhir
        model : str
            Nama model yang akan digunakan
        initial_investment : float
            Jumlah investasi awal
        strategy : str, optional
            Nama strategi trading yang akan dioptimasi
        param_ranges : dict, optional
            Range parameter untuk optimasi
        """
        super().__init__()
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model = model
        self.initial_investment = initial_investment
        self.strategy = strategy
        self.param_ranges = param_ranges
        
    def run(self):
        """Jalankan proses optimasi strategi"""
        try:
            # Pastikan numpy diimpor dengan benar
            import numpy as np
            
            # Prepare data (load ticker data dan prediksi)
            self.progress.emit(10)
            
            # Import prediktor
            from src.models.predictor import StockPredictor
            
            # Buat prediktor
            predictor = StockPredictor(
                ticker=self.ticker,
                model_type=self.model,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Persiapkan data dan latih model
            if not predictor.prepare_data():
                self.error.emit("Error preparing data")
                return
                
            # Latih model
            self.progress.emit(30)
            predictor.train_model()
            
            # Dapatkan prediksi
            actual_prices, predicted_prices, _ = predictor.predict()
            
            # Pastikan dimensi sesuai untuk inverse_transform
            X_shape = predictor.X.shape
            
            # Cek dimensi data untuk menentukan cara transformasi
            if len(X_shape) > 2:  # Multi-feature case
                # Inverse transform untuk mendapatkan harga asli (bukan skala 0-1)
                actual_reshaped = actual_prices.reshape(-1, 1)
                zeros_actual = np.zeros((len(actual_prices), X_shape[2]-1))
                
                actual_prices = predictor.scaler.inverse_transform(
                    np.hstack([actual_reshaped, zeros_actual])
                )[:, 0]
                
                predicted_prices = predictor.scaler.inverse_transform(
                    np.hstack([predicted_prices.reshape(-1, 1), np.zeros((len(predicted_prices), X_shape[2]-1))])
                )[:, 0]
            else:  # Single-feature case (no need for concatenation)
                actual_prices = predictor.scaler.inverse_transform(actual_prices.reshape(-1, 1))[:, 0]
                predicted_prices = predictor.scaler.inverse_transform(predicted_prices.reshape(-1, 1))[:, 0]
            
            self.progress.emit(50)
            
            # Khusus untuk strategi PPO, kita menggunakan implementasi optimize yang berbeda
            if self.strategy.lower() == 'ppo':
                # Untuk PPO, kita akan menjalankan grid search atas parameter
                from src.trading.ppo_agent import PPOTrader
                
                # Setup parameter ranges
                episodes_min = self.param_ranges.get('episodes', {}).get('min', 5)
                episodes_max = self.param_ranges.get('episodes', {}).get('max', 20)
                episodes_step = self.param_ranges.get('episodes', {}).get('step', 5)
                
                gamma_min = self.param_ranges.get('gamma', {}).get('min', 0.95)
                gamma_max = self.param_ranges.get('gamma', {}).get('max', 0.99)
                gamma_step = self.param_ranges.get('gamma', {}).get('step', 0.01)
                
                lambda_min = self.param_ranges.get('lambda', {}).get('min', 0.9)
                lambda_max = self.param_ranges.get('lambda', {}).get('max', 0.99)
                lambda_step = self.param_ranges.get('lambda', {}).get('step', 0.03)
                
                # Generate parameter combinations
                episodes_list = list(range(episodes_min, episodes_max + 1, episodes_step))
                gammas = np.arange(gamma_min, gamma_max + gamma_step, gamma_step).round(3).tolist()
                lambdas = np.arange(lambda_min, lambda_max + lambda_step, lambda_step).round(3).tolist()
                
                # Buat feature untuk PPO
                returns = np.zeros_like(actual_prices)
                returns[1:] = (actual_prices[1:] / actual_prices[:-1]) - 1
                volatility = np.zeros_like(actual_prices)
                window = min(10, len(actual_prices) - 1)
                for i in range(window, len(returns)):
                    volatility[i] = np.std(returns[i-window+1:i+1])
                
                # Pastikan features memiliki dimensi yang benar (2D)
                returns_2d = returns.reshape(-1, 1)
                volatility_2d = volatility.reshape(-1, 1)
                features = np.hstack((returns_2d, volatility_2d))
                
                # Tambahkan fitur prediksi jika tersedia
                if predicted_prices is not None:
                    # Pastikan prediksi memiliki dimensi yang benar
                    pred_col = np.array(predicted_prices).reshape(-1, 1)
                    if len(pred_col) == len(features):
                        features = np.hstack((features, pred_col))
                
                # Grid search
                best_return = -float('inf')
                best_params = None
                best_performance = None
                best_portfolio = None
                best_trades = None
                
                # Hitung total kombinasi untuk progress bar
                total_combinations = len(episodes_list) * len(gammas) * len(lambdas)
                current_combination = 0
                
                for episodes in episodes_list:
                    for gamma in gammas:
                        for lam in lambdas:
                            # Update progress
                            current_combination += 1
                            progress = 50 + int((current_combination / total_combinations) * 40)
                            self.progress.emit(progress)
                            
                            try:
                                # Buat PPO Trader dengan parameter saat ini
                                ppo_trader = PPOTrader(
                                    prices=actual_prices,
                                    features=features,
                                    initial_investment=self.initial_investment
                                )
                                
                                # Sesuaikan parameter
                                ppo_trader.agent.gamma = gamma
                                ppo_trader.agent.lam = lam
                                
                                # Latih model
                                ppo_trader.train(episodes=episodes)
                                
                                # Backtest dengan model terlatih
                                backtest_results = ppo_trader.backtest()
                                
                                # Check if this is the best result so far
                                if backtest_results['performance']['total_return'] > best_return:
                                    best_return = backtest_results['performance']['total_return']
                                    best_params = {'episodes': episodes, 'gamma': gamma, 'lambda': lam}
                                    best_performance = backtest_results['performance']
                                    best_portfolio = backtest_results['portfolio_values']
                                    best_trades = backtest_results['trades']
                            except Exception as e:
                                print(f"Error during PPO optimization with params {episodes}, {gamma}, {lam}: {e}")
                                continue
                
                # Prepare the result
                if best_params is not None:
                    results = [{
                        'strategy': self.strategy,
                        'params': best_params,
                        'initial_investment': best_performance['initial_investment'],
                        'final_value': best_performance['final_value'],
                        'total_return': best_performance['total_return'],
                        'max_drawdown': best_performance['max_drawdown'],
                        'win_rate': best_performance['win_rate'],
                        'num_trades': best_performance['num_trades'],
                        'portfolio_values': best_portfolio,
                        'trades': best_trades
                    }]
                    
                    self.progress.emit(100)
                    self.finished.emit(results)
                    return
                else:
                    # Fallback if no valid combination found
                    self.error.emit("Tidak ada kombinasi parameter PPO yang valid ditemukan")
                    return
            else:
                # Untuk strategi lain, kita menggunakan pendekatan standar
                from src.trading.optimizer import StrategyOptimizer
                optimizer = StrategyOptimizer(actual_prices, predicted_prices, self.initial_investment)
                best_params, best_performance, _, _ = optimizer.optimize(self.strategy, self.param_ranges)
                
                # Run backtest with different parameters
                self.progress.emit(80)
                
                results = []
                total_combinations = 1
                current_combination = 0
                
                # Dapatkan total kombinasi parameter
                for param_values in self.param_ranges.values():
                    total_combinations *= len(param_values)
                
                progress_increment = 20 / total_combinations  # 20% untuk pengujian parameter
                
                # Uji setiap kombinasi parameter
                import itertools
                param_names = list(self.param_ranges.keys())
                param_values = [self.param_ranges[name] for name in param_names]
                
                for param_combo in itertools.product(*param_values):
                    params = dict(zip(param_names, param_combo))
                    
                    # Jalankan backtest dengan parameter ini
                    portfolio_values, trades, performance = self.run_backtest_with_params(
                        actual_prices, predicted_prices, self.initial_investment, self.strategy, params
                    )
                    
                    results.append({
                        'strategy': self.strategy,
                        'params': params,
                        'initial_investment': performance['initial_investment'],
                        'final_value': performance['final_value'],
                        'total_return': performance['total_return'],
                        'max_drawdown': performance['max_drawdown'],
                        'win_rate': performance['win_rate'],
                        'num_trades': performance['num_trades']
                    })
                    
                    current_combination += 1
                    self.progress.emit(80 + int(current_combination * progress_increment))
                
                # Sort results by total_return
                results.sort(key=lambda x: x['total_return'], reverse=True)
                
                self.progress.emit(100)
                
                # Emit results
                self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
            
    def run_backtest_with_params(self, actual_prices, predicted_prices, initial_investment, strategy, params):
        """Implementasi backtest dengan parameter tertentu"""
        # Inisialisasi portfolio
        cash = initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        # Iterasi melalui harga historis
        for i in range(1, len(actual_prices)):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * actual_prices[i]
            portfolio_values.append(portfolio_value)
            
            # Generate signal dengan parameter
            signal = self.generate_signal_with_params(predicted_prices, actual_prices, i, strategy, params)
            
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
        # Menghitung return total
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Menghitung drawdown
        peak = portfolio_values[0] if portfolio_values else initial_investment
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
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
            'initial_investment': initial_investment,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance

    def generate_signal_with_params(self, predicted_prices, actual_prices, index, strategy, params):
        """Generate signal berdasarkan strategi dengan parameter tertentu"""
        from src.trading.strategies import TradingStrategy
        
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        return strategy_function(predicted_prices, actual_prices, index, params)

class ForecastTradeWorker(QThread):
    """Thread worker untuk menjalankan simulasi trading pada data forecast"""
    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, forecast_prices, initial_investment, strategy, params):
        """
        Inisialisasi worker thread
        
        Parameters:
        -----------
        forecast_prices : array-like
            Array harga forecast
        initial_investment : float
            Jumlah investasi awal
        strategy : str
            Nama strategi trading yang akan digunakan
        params : dict
            Parameter untuk strategi
        """
        super().__init__()
        self.forecast_prices = forecast_prices
        self.initial_investment = initial_investment
        self.strategy = strategy
        self.params = params
        
    def run(self):
        """Jalankan proses simulasi trading"""
        try:
            # Simulasi trading dengan forecast
            self.progress.emit(20)
            
            portfolio_values, trades, performance = self.simulate_forecast_trading(
                self.forecast_prices, self.initial_investment, self.strategy, self.params
            )
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(portfolio_values, trades, performance)
        except Exception as e:
            self.error.emit(str(e))
    
    def simulate_forecast_trading(self, forecast_prices, initial_investment, strategy, params):
        """Implementasi simulasi trading pada data forecast"""
        # Inisialisasi portfolio
        cash = initial_investment
        shares = 0
        portfolio_values = []
        trades = []
        
        from src.trading.strategies import TradingStrategy
        strategy_function = TradingStrategy.get_strategy_function(strategy)
        
        # Jika strategi PPO, latih agen terlebih dahulu
        if strategy == 'ppo':
            from src.trading.ppo_agent import PPOTrader, TradingEnv
            import numpy as np
            
            # Ambil parameter
            episodes = params.get('episodes', 10)
            gamma = params.get('gamma', 0.99)
            lam = params.get('lambda', 0.95)
            
            # Siapkan fitur tambahan (volatilitas simple)
            forecast_returns = np.zeros_like(forecast_prices)
            forecast_returns[1:] = (forecast_prices[1:] / forecast_prices[:-1]) - 1
            volatility = np.zeros_like(forecast_prices)
            window = min(10, len(forecast_prices) - 1)
            for i in range(window, len(forecast_returns)):
                volatility[i] = np.std(forecast_returns[i-window+1:i+1])
            
            # Pastikan dimensi array benar sebelum menggabungkan
            returns_2d = forecast_returns.reshape(-1, 1)
            volatility_2d = volatility.reshape(-1, 1)
            features = np.hstack((returns_2d, volatility_2d))
            
            # Buat dan latih PPO agent
            ppo_trader = PPOTrader(
                prices=forecast_prices,
                features=features,
                initial_investment=initial_investment
            )
            
            # Sesuaikan parameter PPO agent
            ppo_trader.agent.gamma = gamma
            ppo_trader.agent.lam = lam
            
            # Latih model
            print(f"Melatih PPO Agent dengan {episodes} episodes...")
            try:
                train_results = ppo_trader.train(episodes=episodes)
                
                # Gunakan agent yang sudah dilatih untuk trading
                backtest_results = ppo_trader.backtest()
                
                return (
                    backtest_results['portfolio_values'],
                    backtest_results['trades'],
                    backtest_results['performance']
                )
            except Exception as e:
                print(f"Error melatih PPO agent: {e}")
                # Jika gagal, gunakan strategi predictive sebagai fallback
                strategy = 'predictive'
                strategy_function = TradingStrategy.get_strategy_function(strategy)
                params = {'buy_threshold': 1.01, 'sell_threshold': 0.99}
        
        # Untuk strategi non-PPO, gunakan implementasi yang sudah ada
        # Karena kita hanya memiliki data forecast, kita akan menggunakan sinyal prediktif
        # yang langsung dibandingkan dengan harga berikutnya
        for i in range(len(forecast_prices) - 1):
            # Hitung nilai portfolio saat ini
            portfolio_value = cash + shares * forecast_prices[i]
            portfolio_values.append(portfolio_value)
            
            # Generate signal
            current_price = forecast_prices[i]
            next_price = forecast_prices[i + 1]
            
            # Implementasi sederhana: anggap next_price sebagai "prediksi"
            signal = 'HOLD'
            
            if strategy == 'trend_following':
                threshold = params.get('threshold', 0.01)
                if next_price > current_price * (1 + threshold):
                    signal = 'BUY'
                elif next_price < current_price * (1 - threshold):
                    signal = 'SELL'
            
            elif strategy == 'mean_reversion':
                # Untuk mean reversion, kita perlu membangun rata-rata bergerak
                # dari data forecast sejauh ini
                window = params.get('window', 10)
                buy_threshold = params.get('buy_threshold', 0.05)
                sell_threshold = params.get('sell_threshold', 0.05)
                
                if i + 1 >= window:
                    moving_avg = np.mean(forecast_prices[i+1-window:i+1])
                    if current_price < moving_avg * (1 - buy_threshold):
                        signal = 'BUY'  # Beli ketika harga di bawah rata-rata
                    elif current_price > moving_avg * (1 + sell_threshold):
                        signal = 'SELL'  # Jual ketika harga di atas rata-rata
            
            elif strategy == 'predictive':
                buy_threshold = params.get('buy_threshold', 1.01)
                sell_threshold = params.get('sell_threshold', 0.99)
                
                if next_price > current_price * buy_threshold:
                    signal = 'BUY'
                elif next_price < current_price * sell_threshold:
                    signal = 'SELL'
            
            # Proses signals
            if signal == 'BUY' and cash > 0:
                # Beli saham sebanyak mungkin dengan uang yang ada
                shares_to_buy = cash / current_price
                shares += shares_to_buy
                cash = 0
                trades.append({
                    'day': i,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'value': shares_to_buy * current_price
                })
            elif signal == 'SELL' and shares > 0:
                # Jual semua saham
                cash += shares * current_price
                trades.append({
                    'day': i,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'value': shares * current_price
                })
                shares = 0
        
        # Tambahkan nilai akhir portfolio
        if len(forecast_prices) > 0:
            portfolio_values.append(cash + shares * forecast_prices[-1])
        
        # Nilai akhir portfolio
        final_value = cash + (shares * forecast_prices[-1] if len(forecast_prices) > 0 else 0)
        
        # Menghitung metrik performa
        # Menghitung return total
        total_return = (final_value - initial_investment) / initial_investment * 100
        
        # Menghitung drawdown
        peak = portfolio_values[0] if portfolio_values else initial_investment
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
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
            'initial_investment': initial_investment,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return portfolio_values, trades, performance 