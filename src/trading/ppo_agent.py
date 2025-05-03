"""
PPO Agent Module
=============

Modul ini berisi implementasi PPO (Proximal Policy Optimization) agent
untuk optimasi strategi trading menggunakan reinforcement learning.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import gym
from gym import spaces

class Buffer:
    """
    Buffer untuk menyimpan experience untuk agen PPO
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def clear(self):
        """Bersihkan buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

class TradingEnv(gym.Env):
    """
    Trading environment yang kompatibel dengan gym untuk reinforcement learning
    """
    def __init__(self, prices, features=None, initial_balance=10000, transaction_fee=0.001):
        """
        Inisialisasi trading environment
        
        Parameters:
        -----------
        prices : array-like
            Data harga historis
        features : array-like, optional
            Fitur tambahan untuk model (prediksi, indikator teknikal, dll)
        initial_balance : float, optional
            Saldo awal yang tersedia, default 10000
        transaction_fee : float, optional
            Biaya transaksi sebagai fraksi dari nilai transaksi, default 0.001 (0.1%)
        """
        super(TradingEnv, self).__init__()
        
        self.prices = prices
        self.features = features if features is not None else np.zeros((len(prices), 1))
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # State shape: [balance, owned_shares, current_price, position_value, unrealized_pnl, *features, *technical_indicators]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(8 + self.features.shape[1],)  # 8 state variables + features
        )
        
        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Informasi tambahan untuk trading
        self.avg_entry_price = 0  # Harga rata-rata masuk
        self.position_value = 0   # Nilai posisi saat ini
        self.unrealized_pnl = 0   # P&L yang belum direalisasi
        
        # Reset environment
        self.reset()
        
    def reset(self):
        """Reset environment ke keadaan awal"""
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        self.done = False
        self.portfolio_value_history = [self.initial_balance]
        self.avg_entry_price = 0
        self.position_value = 0
        self.unrealized_pnl = 0
        
        return self._get_observation()
        
    def step(self, action):
        """
        Melakukan langkah berdasarkan aksi yang diberikan
        
        Parameters:
        -----------
        action : int
            0 = hold, 1 = buy, 2 = sell
            
        Returns:
        --------
        tuple
            (observation, reward, done, info)
        """
        if self.done:
            return self._get_observation(), 0, self.done, {}
            
        current_price = self.prices[self.current_step]
        prev_portfolio_value = self.balance + self.shares * current_price
        
        # Menjalankan aksi
        if action == 1:  # Buy
            max_shares = self.balance / (current_price * (1 + self.transaction_fee))
            new_shares = max_shares
            cost = new_shares * current_price * (1 + self.transaction_fee)
            if cost > 0:
                # Update harga rata-rata masuk jika sudah memiliki saham
                if self.shares > 0:
                    self.avg_entry_price = (self.shares * self.avg_entry_price + new_shares * current_price) / (self.shares + new_shares)
                else:
                    self.avg_entry_price = current_price
                
                self.shares += new_shares
                self.balance -= cost
        elif action == 2:  # Sell
            if self.shares > 0:
                sell_value = self.shares * current_price * (1 - self.transaction_fee)
                self.balance += sell_value
                self.shares = 0
                self.avg_entry_price = 0
        
        # Pindah ke langkah berikutnya
        self.current_step += 1
        
        # Cek apakah episode selesai
        if self.current_step >= len(self.prices) - 1:
            self.done = True
            
        # Hitung nilai portfolio baru
        new_price = self.prices[self.current_step]
        self.position_value = self.shares * new_price
        new_portfolio_value = self.balance + self.position_value
        
        # Update unrealized P&L
        if self.shares > 0 and self.avg_entry_price > 0:
            self.unrealized_pnl = (new_price - self.avg_entry_price) * self.shares
        else:
            self.unrealized_pnl = 0
            
        self.portfolio_value_history.append(new_portfolio_value)
        
        # Hitung reward (return dari langkah sebelumnya)
        reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Tambahkan bonus reward untuk nilai akhir portfolio yang lebih tinggi
        if self.done:
            final_return = (new_portfolio_value - self.initial_balance) / self.initial_balance
            reward += final_return
            
        return self._get_observation(), reward, self.done, {
            'portfolio_value': new_portfolio_value,
            'balance': self.balance,
            'shares': self.shares,
            'unrealized_pnl': self.unrealized_pnl,
            'position_value': self.position_value
        }
        
    def _get_observation(self):
        """
        Dapatkan observasi untuk kondisi saat ini dengan fitur yang diperkaya
        """
        current_price = self.prices[self.current_step]
        
        # Indikator teknikal
        if self.current_step >= 20:
            # SMA (Simple Moving Average)
            sma_short = np.mean(self.prices[self.current_step-5:self.current_step+1])
            sma_long = np.mean(self.prices[self.current_step-20:self.current_step+1])
            
            # Rasio harga saat ini terhadap SMA (relatif ke 1)
            price_sma_ratio_short = current_price / sma_short - 1
            price_sma_ratio_long = current_price / sma_long - 1
            
            # Volatilitas (diukur sebagai standar deviasi return dalam 20 hari)
            # Perbaikan - gunakan window lokal untuk memastikan dimensi array yang dibagi sesuai
            price_window = self.prices[self.current_step-20:self.current_step+1]
            returns = np.diff(price_window) / price_window[:-1]
            volatility = np.std(returns)
            
            # Momentum (perubahan harga dalam periode tertentu)
            momentum_5d = self.prices[self.current_step] / self.prices[self.current_step-5] - 1 if self.current_step >= 5 else 0
        else:
            # Default nilai untuk awal episode
            sma_short = sma_long = current_price
            price_sma_ratio_short = price_sma_ratio_long = 0
            volatility = 0.02  # Default volatilitas
            momentum_5d = 0
            
        # Informasi portofolio
        portfolio_value = self.balance + self.position_value
        position_ratio = self.position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Ambil features dari timestep saat ini
        try:
            current_features = self.features[self.current_step]
        except (IndexError, TypeError) as e:
            print(f"Warning: Tidak bisa mengambil features: {e}")
            current_features = np.zeros(self.features.shape[1], dtype=np.float32)
        
        # Pastikan current_features adalah numpy array
        if not isinstance(current_features, np.ndarray):
            if isinstance(current_features, (int, float)):
                current_features = np.array([current_features], dtype=np.float32)
            elif isinstance(current_features, list):
                current_features = np.array(current_features, dtype=np.float32)
            else:
                try:
                    current_features = np.array(current_features, dtype=np.float32)
                except:
                    print(f"Warning: Tidak bisa mengkonversi features ke array: {type(current_features)}")
                    current_features = np.zeros(self.features.shape[1], dtype=np.float32)
        
        # Pastikan current_features adalah 1D array
        if len(current_features.shape) > 1:
            current_features = current_features.flatten()
        
        # Pastikan tipe data konsisten
        current_features = current_features.astype(np.float32)
        
        # State dasar dengan fitur diperkaya
        base_info = np.array([
            self.balance / self.initial_balance,  # Normalisasi balance
            self.shares * current_price / self.initial_balance,  # Posisi relatif terhadap modal awal
            current_price,
            self.position_value / self.initial_balance,  # Posisi relatif
            self.unrealized_pnl / self.initial_balance,  # PnL relatif
            price_sma_ratio_short,  # Relatif ke SMA jangka pendek
            price_sma_ratio_long,   # Relatif ke SMA jangka panjang
            volatility              # Volatilitas
        ], dtype=np.float32)
        
        # Coba gabungkan arrays
        try:
            obs = np.concatenate([base_info, current_features])
        except Exception as e:
            print(f"Warning: Tidak bisa menggabungkan arrays: {e}, base_info.shape={base_info.shape}, features.shape={current_features.shape}")
            obs = base_info
        
        # Simpan state_dim untuk PPO Agent
        if not hasattr(self, 'state_dimensions_calculated') or not self.state_dimensions_calculated:
            self.state_dim = len(obs)
            self.state_dimensions_calculated = True
        
        return obs
        
class PPOAgent:
    """
    Agent yang mengimplementasikan algoritma PPO untuk trading
    """
    def __init__(
        self, 
        state_dim, 
        action_dim,
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        clip_ratio=0.2,
        batch_size=64,
        epochs=10,
        lam=0.95
    ):
        """
        Inisialisasi PPO Agent
        
        Parameters:
        -----------
        state_dim : int
            Dimensi dari state space
        action_dim : int
            Dimensi dari action space
        actor_lr : float, optional
            Learning rate untuk actor network
        critic_lr : float, optional
            Learning rate untuk critic network
        gamma : float, optional
            Discount factor untuk reward
        clip_ratio : float, optional
            Batasan perubahan policy dalam PPO
        batch_size : int, optional
            Ukuran batch untuk training
        epochs : int, optional
            Jumlah epoch training per batch
        lam : float, optional
            Parameter lambda untuk Generalized Advantage Estimation (GAE)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.lam = lam
        
        # Build and compile actor-critic model
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Initialize buffer
        self.buffer = Buffer()
        
    def _build_actor(self):
        """Buat actor network"""
        state_input = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        output = Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=state_input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.actor_lr))
        return model
        
    def _build_critic(self):
        """Buat critic network"""
        state_input = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=state_input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.critic_lr), loss='mse')
        return model
        
    def get_action(self, state):
        """
        Memilih aksi berdasarkan state saat ini
        
        Parameters:
        -----------
        state : array-like
            State saat ini
            
        Returns:
        --------
        tuple
            (aksi, log probabilitas, nilai state)
        """
        # Pastikan state dalam bentuk numpy array
        if not isinstance(state, np.ndarray):
            try:
                state = np.array(state, dtype=np.float32)
            except Exception as e:
                print(f"Error converting state to numpy array: {e}")
                # Jika gagal, gunakan array nol sebagai fallback
                state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Pastikan dimensi state sesuai
        if len(state.shape) == 0:  # Scalar
            state = np.array([float(state)], dtype=np.float32)
        
        if len(state.shape) == 1:  # Vector 1D
            # Pastikan panjang state sesuai dengan yang diharapkan
            if state.shape[0] != self.state_dim:
                # Jika tidak sesuai, sesuaikan ukurannya
                if state.shape[0] > self.state_dim:
                    # Potong jika terlalu panjang
                    state = state[:self.state_dim]
                else:
                    # Pad dengan nol jika terlalu pendek
                    padding = np.zeros(self.state_dim - state.shape[0], dtype=np.float32)
                    state = np.concatenate([state, padding])
            
            # Reshape untuk batch prediction (1, state_dim)
            state = state.reshape(1, -1)
        
        # Prediksi action probabilities dan value
        try:
            action_probs = self.actor.predict(state, verbose=0)[0]
            value = self.critic.predict(state, verbose=0)[0][0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Jika prediksi gagal, gunakan distribusi uniform dan nilai 0
            action_probs = np.ones(self.action_dim) / self.action_dim
            value = 0.0
        
        # Pilih aksi berdasarkan probabilitas
        try:
            action = np.random.choice(self.action_dim, p=action_probs)
            log_prob = np.log(action_probs[action] + 1e-10)  # Hindari log(0)
        except Exception as e:
            print(f"Error selecting action: {e}")
            # Jika pemilihan aksi gagal, pilih aksi acak
            action = np.random.randint(0, self.action_dim)
            log_prob = -np.log(self.action_dim)  # Log dari uniform probability
        
        return int(action), float(log_prob), float(value)
        
    def store_transition(self, state, action, reward, value, log_prob, done):
        """
        Simpan transisi ke buffer
        
        Parameters:
        -----------
        state : array
            State saat ini
        action : int
            Action yang dilakukan
        reward : float
            Reward yang didapat
        value : float
            Nilai state saat ini (dari critic)
        log_prob : float
            Log probabilitas dari action
        done : bool
            Flag jika episode sudah selesai
        """
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.values.append(value)
        self.buffer.log_probs.append(log_prob)
        self.buffer.dones.append(done)
        
    def clear_buffer(self):
        """Bersihkan buffer"""
        self.buffer.clear()
        
    def calculate_advantages(self):
        """Menghitung advantage untuk setiap transisi di buffer menggunakan GAE"""
        returns = np.zeros_like(self.buffer.rewards)
        advantages = np.zeros_like(self.buffer.rewards)
        
        next_value = 0
        next_advantage = 0
        
        # Hitung returns dan advantages dari belakang (backward)
        for t in reversed(range(len(self.buffer.rewards))):
            # Hitung bootstrapped return
            returns[t] = self.buffer.rewards[t] + self.gamma * next_value * (1 - self.buffer.dones[t])
            
            # Hitung TD error
            td_error = self.buffer.rewards[t] + self.gamma * next_value * (1 - self.buffer.dones[t]) - self.buffer.values[t]
            
            # Generalized Advantage Estimation (GAE)
            advantages[t] = td_error + self.gamma * self.lam * next_advantage * (1 - self.buffer.dones[t])
            
            next_value = self.buffer.values[t]
            next_advantage = advantages[t]
        
        return returns, advantages
        
    def train(self, batch_size=64):
        """
        Latih PPO agent dengan data dari buffer
        
        Parameters:
        -----------
        batch_size : int
            Ukuran batch untuk training
        """
        # Konversi buffer ke numpy arrays
        states = np.array(self.buffer.states, dtype=np.float32)
        actions = np.array(self.buffer.actions, dtype=np.int32)
        old_log_probs = np.array(self.buffer.log_probs, dtype=np.float32)
        
        # Hitung returns dan advantages
        returns, advantages = self.calculate_advantages()
        
        # Normalisasi advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch training
        indices = np.arange(len(states))
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                if end > len(indices):
                    end = len(indices)
                
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Latih actor dan critic dengan mini-batch
                self._update_actor_critic(
                    batch_states, 
                    batch_actions, 
                    batch_old_log_probs, 
                    batch_returns, 
                    batch_advantages
                )
        
        # Setelah training selesai, kosongkan buffer
        self.buffer.clear()

    def _update_actor_critic(self, states, actions, old_log_probs, returns, advantages):
        """
        Update parameter actor dan critic dengan backpropagation
        
        Parameters:
        -----------
        states : numpy.ndarray
            Batch state observations
        actions : numpy.ndarray
            Batch actions yang diambil
        old_log_probs : numpy.ndarray
            Log probabilitas dari aksi dengan policy lama
        returns : numpy.ndarray
            Expected returns (untuk critic)
        advantages : numpy.ndarray
            Advantage estimates (untuk actor)
        """
        # Update critic
        with tf.GradientTape() as tape:
            values = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
        
        # Hitung gradients dan update weights critic
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Update actor
        with tf.GradientTape() as tape:
            # Prediksi probabilitas aksi
            action_probs = self.actor(states, training=True)
            
            # Ambil probabilitas untuk aksi yang diambil
            indices = tf.stack([
                tf.range(tf.shape(actions)[0]), 
                tf.cast(actions, tf.int32)
            ], axis=1)
            
            action_prob = tf.gather_nd(action_probs, indices)
            new_log_probs = tf.math.log(action_prob + 1e-10)
            
            # Hitung ratio untuk clipped objective
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            # Hitung surrogate losses
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # PPO loss (negatif karena kita ingin maximize)
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        
        # Hitung gradients dan update weights actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return {
            'actor_loss': actor_loss.numpy(),
            'critic_loss': critic_loss.numpy()
        }

class PPOTrader:
    """
    Wrapper class untuk menerapkan PPO dalam konteks trading
    """
    def __init__(self, prices, features=None, initial_investment=10000):
        """
        Inisialisasi PPO Trader
        
        Parameters:
        -----------
        prices : array-like
            Data harga historis
        features : array-like, optional
            Fitur tambahan (termasuk prediksi model)
        initial_investment : float, optional
            Jumlah investasi awal, default 10000
        """
        # Setup environment
        if features is None:
            # Jika tidak ada features, gunakan return dan volatilitas sebagai feature dasar
            returns = np.zeros_like(prices)
            returns[1:] = (prices[1:] / prices[:-1]) - 1
            
            # Volatilitas (standar deviasi return rolling window)
            volatility = np.zeros_like(prices)
            window = 10
            for i in range(window, len(returns)):
                volatility[i] = np.std(returns[i-window+1:i+1])
                
            features = np.column_stack((returns, volatility))
            
        self.env = TradingEnv(prices, features, initial_investment)
        
        # Setup agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.agent = PPOAgent(state_dim, action_dim)
        
        self.trained = False
        
    def train(self, episodes=100, max_steps=None):
        """
        Melatih agent PPO
        
        Parameters:
        -----------
        episodes : int, optional
            Jumlah episode training, default 100
        max_steps : int, optional
            Maksimum langkah per episode, default None (gunakan seluruh dataset)
            
        Returns:
        --------
        dict
            Dictionary berisi hasil training
        """
        if max_steps is None:
            max_steps = len(self.env.prices)
            
        best_reward = -np.inf
        episode_rewards = []
        portfolio_values = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                action, log_prob, value = self.agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.store_transition(
                    state, action, reward, value, log_prob, done
                )
                
                state = next_state
                episode_reward += reward
                step += 1
                
            # Dapatkan nilai estimasi state terakhir
            _, _, last_value = self.agent.get_action(state)
            
            # Train the agent
            self.agent.train()
            
            episode_rewards.append(episode_reward)
            portfolio_values.append(self.env.portfolio_value_history)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f}, "
                      f"Portfolio Value: {self.env.portfolio_value_history[-1]:.2f}")
                      
        self.trained = True
        
        return {
            'episode_rewards': episode_rewards,
            'portfolio_values': portfolio_values,
            'best_reward': best_reward
        }
        
    def backtest(self, prices=None, features=None, initial_investment=None):
        """
        Melakukan backtest agent yang telah dilatih
        
        Parameters:
        -----------
        prices : array-like, optional
            Data harga untuk backtest, jika None menggunakan data training
        features : array-like, optional
            Fitur untuk backtest, jika None menggunakan data training
        initial_investment : float, optional
            Jumlah investasi awal untuk backtest
            
        Returns:
        --------
        dict
            Dictionary berisi hasil backtest
        """
        # Gunakan data baru jika disediakan, atau kembali ke data training
        if prices is not None or features is not None or initial_investment is not None:
            test_prices = prices if prices is not None else self.env.prices
            test_features = features if features is not None else self.env.features
            test_initial = initial_investment if initial_investment is not None else self.env.initial_balance
            
            test_env = TradingEnv(test_prices, test_features, test_initial)
        else:
            test_env = self.env
            test_env.reset()
            
        state = test_env.reset()
        done = False
        trades = []
        actions_taken = []
        
        while not done:
            # Gunakan model untuk memilih aksi
            action, _, _ = self.agent.get_action(state)
            actions_taken.append(action)
            
            # Harga saat ini sebelum mengambil langkah
            current_price = test_env.prices[test_env.current_step]
            current_shares = test_env.shares
            current_balance = test_env.balance
            current_step = test_env.current_step
            
            # Ambil langkah
            next_state, reward, done, info = test_env.step(action)
            
            # Rekam trade jika terjadi
            if action == 1 and info['shares'] > current_shares:  # BUY
                trades.append({
                    'day': current_step,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': info['shares'] - current_shares,
                    'value': (info['shares'] - current_shares) * current_price
                })
            elif action == 2 and info['shares'] < current_shares:  # SELL
                trades.append({
                    'day': current_step,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': current_shares - info['shares'],
                    'value': (current_shares - info['shares']) * current_price
                })
                
            state = next_state
            
        # Calculate performance metrics
        portfolio_values = test_env.portfolio_value_history
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculate max drawdown
        peak = portfolio_values[0]
        drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown = max(drawdown, dd)
        
        # Calculate Sharpe ratio
        if len(portfolio_values) > 1:
            daily_returns = [(portfolio_values[i]/portfolio_values[i-1])-1 for i in range(1, len(portfolio_values))]
            sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Count win/loss trades
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
            'initial_investment': test_env.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
        
        return {
            'portfolio_values': portfolio_values,
            'trades': trades,
            'performance': performance,
            'actions': actions_taken
        } 