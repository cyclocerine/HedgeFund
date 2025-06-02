"""
PPO Agent
========

Implementasi Proximal Policy Optimization (PPO) untuk menghasilkan sinyal trading.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
import random
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        
        # Shared features dengan layer yang lebih dalam
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Actor head dengan temperature scaling
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # Critic head dengan lebih banyak layer
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1)
        )
        
        # Initialize weights dengan gain yang lebih tinggi
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.5))
            module.bias.data.zero_()
    
    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(next(self.parameters()).device)
        features = self.features(state)
        
        # Apply temperature scaling untuk meningkatkan confidence
        logits = self.actor(features)
        action_probs = F.softmax(logits / self.temperature, dim=-1)
        
        value = self.critic(features)
        return action_probs, value

class PPOAgent:
    def __init__(self, state_dim, action_dim=3, hidden_dim=128, lr=3e-4, gamma=0.99, epsilon=0.2, c1=1, c2=0.01):
        """
        Inisialisasi PPO Agent dengan parameter yang dioptimalkan
        
        Parameters:
        -----------
        state_dim : int
            Dimensi state (fitur input)
        action_dim : int
            Jumlah aksi yang mungkin (default: 3 untuk buy, sell, hold)
        hidden_dim : int
            Dimensi hidden layer (ditingkatkan dari 64 ke 128)
        lr : float
            Learning rate
        gamma : float
            Discount factor
        epsilon : float
            PPO clip parameter
        c1 : float
            Value loss coefficient
        c2 : float
            Entropy coefficient untuk mendorong eksplorasi
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': [p for n, p in self.actor_critic.named_parameters() if 'temperature' not in n]},
            {'params': [self.actor_critic.temperature], 'lr': lr * 0.1}  # Lower learning rate for temperature
        ], lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        
        # Tambahkan memory untuk experience replay
        self.memory_size = 1000
        self.memory = []
    
    def get_action(self, state):
        """
        Mendapatkan aksi berdasarkan state dengan confidence yang lebih baik
        
        Returns:
        --------
        action : int
            Indeks aksi (0: buy, 1: sell, 2: hold)
        confidence : float
            Tingkat keyakinan (probability) dari aksi yang dipilih
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
            
            # Gunakan temperature untuk mengatur sharpness dari distribusi
            temperature = self.actor_critic.temperature.item()
            scaled_probs = action_probs ** (1 / temperature)
            scaled_probs = scaled_probs / scaled_probs.sum()
            
            # Sample action
            dist = Categorical(scaled_probs)
            action = dist.sample()
            
            # Get confidence (probability) of the selected action
            confidence = scaled_probs[action.item()].item()
            
            return action.item(), confidence
    
    def train(self, states, n_epochs=50, batch_size=32):
        """
        Melatih agent PPO dengan experience replay dan curriculum learning
        
        Parameters:
        -----------
        states : numpy.ndarray
            Array dari state-state historis
        n_epochs : int
            Jumlah epoch training
        batch_size : int
            Ukuran batch
        """
        states = torch.FloatTensor(states).to(self.device)
        n_states = len(states)
        
        # Curriculum learning - mulai dengan batch kecil
        curr_batch_size = batch_size // 2
        
        # Inisialisasi reward history untuk adaptive reward scaling
        reward_history = []
        
        for epoch in range(n_epochs):
            # Increase batch size gradually
            if epoch > n_epochs // 2:
                curr_batch_size = batch_size
            
            # Generate batches
            indices = np.random.permutation(n_states)
            total_loss = 0
            n_batches = 0
            
            for start_idx in range(0, n_states, curr_batch_size):
                batch_indices = indices[start_idx:start_idx + curr_batch_size]
                batch_states = states[batch_indices]
                
                # Get old action probabilities and values
                with torch.no_grad():
                    old_action_probs, old_values = self.actor_critic(batch_states)
                    old_actions = Categorical(old_action_probs).sample()
                    old_log_probs = Categorical(old_action_probs).log_prob(old_actions)
                
                # Compute rewards with trend-based incentives
                rewards = torch.zeros(len(batch_indices)).to(self.device)
                for i in range(len(batch_indices)):
                    if i > 0:
                        # Hitung perubahan harga
                        price_change = (batch_states[i][-1] - batch_states[i-1][-1]) / batch_states[i-1][-1]
                        
                        # Reward untuk aksi yang sesuai dengan trend
                        if old_actions[i] == 0:  # Buy
                            rewards[i] = 1.0 if price_change > 0.01 else -0.5
                        elif old_actions[i] == 1:  # Sell
                            rewards[i] = 1.0 if price_change < -0.01 else -0.5
                        else:  # Hold
                            rewards[i] = 0.2 if abs(price_change) < 0.005 else -0.1
                
                # Add experience to memory
                for i, state in enumerate(batch_states):
                    if len(self.memory) >= self.memory_size:
                        self.memory.pop(0)
                    self.memory.append((
                        state, old_actions[i], rewards[i], old_values[i].squeeze()
                    ))
                
                # Sample from memory for experience replay
                if len(self.memory) > batch_size and epoch > n_epochs // 4:
                    replay_size = min(batch_size // 4, len(self.memory))
                    replay_samples = random.sample(self.memory, replay_size)
                    replay_states = torch.stack([s[0] for s in replay_samples])
                    replay_actions = torch.tensor([s[1] for s in replay_samples]).to(self.device)
                    replay_rewards = torch.tensor([s[2] for s in replay_samples]).to(self.device)
                    replay_values = torch.tensor([s[3] for s in replay_samples]).to(self.device)
                    
                    # Add replay samples to current batch
                    batch_states = torch.cat([batch_states, replay_states])
                    old_actions = torch.cat([old_actions, replay_actions])
                    rewards = torch.cat([rewards, replay_rewards])
                    old_values = torch.cat([old_values.squeeze(), replay_values])
                    
                    # Recompute log probs for combined batch
                    with torch.no_grad():
                        combined_probs, _ = self.actor_critic(batch_states)
                        old_log_probs = Categorical(combined_probs).log_prob(old_actions)
                
                # Store rewards for adaptive scaling
                reward_history.extend(rewards.cpu().numpy())
                if len(reward_history) > 1000:
                    reward_history = reward_history[-1000:]
                
                # Compute returns and advantages with adaptive scaling
                returns = rewards.clone()
                if len(reward_history) > 0:
                    reward_std = np.std(reward_history) + 1e-8
                    returns = returns / reward_std
                
                advantages = returns - old_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO update
                for _ in range(5):
                    # Get current action probabilities and values
                    action_probs, values = self.actor_critic(batch_states)
                    dist = Categorical(action_probs)
                    log_probs = dist.log_prob(old_actions)
                    
                    # Compute ratio and clipped ratio
                    ratio = torch.exp(log_probs - old_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
                    
                    # Compute losses with adaptive weights
                    policy_loss = -torch.min(
                        ratio * advantages,
                        clipped_ratio * advantages
                    ).mean() * 1.2  # Slightly increased weight
                    
                    value_loss = self.c1 * ((values.squeeze() - returns) ** 2).mean()
                    
                    # Dynamic entropy weight with minimum threshold
                    current_entropy = dist.entropy().mean()
                    target_entropy = max(0.3, 0.8 * (1 - epoch / n_epochs))  # Higher minimum entropy
                    entropy_weight = self.c2 * max(0.2, 1 - epoch / (n_epochs * 0.9))  # Slower decay
                    entropy_loss = -entropy_weight * current_entropy
                    
                    total_loss = policy_loss + value_loss + entropy_loss
                    
                    # Update parameters
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                    self.optimizer.step()
                    
                    # Adjust temperature with bounds
                    with torch.no_grad():
                        if current_entropy > target_entropy:
                            self.actor_critic.temperature.data *= 0.995  # Slower decrease
                        else:
                            self.actor_critic.temperature.data *= 1.005  # Slower increase
                        
                        # Clamp temperature with higher minimum
                        self.actor_critic.temperature.data.clamp_(0.3, 1.5)
                
                n_batches += 1
                total_loss += total_loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                action_counts = torch.bincount(old_actions, minlength=3)
                action_dist = action_counts.float() / action_counts.sum()
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, "
                      f"Entropy: {current_entropy:.4f}, Temp: {self.actor_critic.temperature.item():.2f}, "
                      f"Actions: Buy={action_dist[0]:.2f}, Sell={action_dist[1]:.2f}, Hold={action_dist[2]:.2f}")
    
    def save(self, path):
        """Menyimpan model"""
        torch.save(self.actor_critic.state_dict(), path)
    
    def load(self, path):
        """Memuat model"""
        self.actor_critic.load_state_dict(torch.load(path))

class TradingEnv(gym.Env):
    """
    Environment untuk trading yang mengimplementasikan Gym interface
    """
    def __init__(self, prices, features=None, initial_balance=10000, transaction_fee=0.001, lookback=60, forecast_days=30):
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
        lookback : int, optional
            Jumlah hari historis yang digunakan untuk prediksi, default 60
        forecast_days : int, optional
            Jumlah hari yang akan diprediksi ke depan, default 30
        """
        super(TradingEnv, self).__init__()
        
        self.prices = np.array(prices)
        self.features = np.array(features) if features is not None else np.zeros((len(prices), 1))
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Hitung fitur dasar
        self.returns = np.zeros_like(prices)
        self.returns[1:] = (prices[1:] / prices[:-1]) - 1
        
        # Volatilitas (standar deviasi return rolling window)
        self.volatility = np.zeros_like(prices)
        window = 10
        for i in range(window, len(self.returns)):
            self.volatility[i] = np.std(self.returns[i-window+1:i+1])
        
        # State space: [price, return, volatility, position, balance] + features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(5 + self.features.shape[1] + lookback,),
            dtype=np.float32
        )
        
        # Action space: 0 = buy, 1 = sell, 2 = hold
        self.action_space = spaces.Discrete(3)
        
        self.lookback = lookback
        self.forecast_days = forecast_days
        
        self.reset()
    
    def reset(self):
        """Reset environment ke kondisi awal"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.position_value = 0
        self.portfolio_values = [self.initial_balance]
        return self._get_state()
    
    def _get_state(self):
        """Mendapatkan state saat ini"""
        price = self.prices[self.current_step]
        ret = self.returns[self.current_step]
        vol = self.volatility[self.current_step]
        position = self.shares * price / self.initial_balance
        balance = self.balance / self.initial_balance
        
        state = np.array([
            price / self.prices[0],  # Normalisasi harga
            ret,
            vol,
            position,
            balance
        ], dtype=np.float32)
        
        # Tambahkan features jika ada
        if self.features is not None:
            state = np.concatenate([state, self.features[self.current_step]])
        
        # Tambahkan data historis
        if self.current_step >= self.lookback:
            state = np.concatenate([state, self.returns[self.current_step-self.lookback:self.current_step]])
        else:
            state = np.concatenate([state, np.zeros(self.lookback-self.current_step)])
        
        # Normalisasi data historis
        state[:self.lookback] = (state[:self.lookback] - state[:self.lookback].mean()) / (state[:self.lookback].std() + 1e-8)
        
        return state
        
    def step(self, action):
        """
        Melakukan langkah trading
        
        Parameters:
        -----------
        action : int
            0: buy, 1: sell, 2: hold
            
        Returns:
        --------
        tuple
            (next_state, reward, done, info)
        """
        assert self.action_space.contains(action), f"Action {action} is invalid"
            
        current_price = self.prices[self.current_step]
        old_portfolio_value = self.balance + self.shares * current_price
        
        # Execute action
        if action == 0:  # Buy
            max_shares = self.balance / (current_price * (1 + self.transaction_fee))
            if max_shares > 0:
                self.shares += max_shares
                self.balance -= max_shares * current_price * (1 + self.transaction_fee)
        elif action == 1:  # Sell
            if self.shares > 0:
                sell_value = self.shares * current_price * (1 - self.transaction_fee)
                self.balance += sell_value
                self.shares = 0
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        if not done:
            new_price = self.prices[self.current_step]
            new_portfolio_value = self.balance + self.shares * new_price
            self.portfolio_values.append(new_portfolio_value)
            
            # Calculate reward
            reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
            
            # Add penalty for frequent trading to encourage holding
            if action != 2:  # If not hold
                reward -= 0.0001  # Small penalty for trading
            
            # Add bonus for profitable trades
            if new_portfolio_value > old_portfolio_value:
                reward += 0.0001  # Small bonus for profitable trades
            
            next_state = self._get_state()
        else:
            # Final reward based on overall performance
            final_return = (self.portfolio_values[-1] - self.initial_balance) / self.initial_balance
            reward = final_return
            next_state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_values[-1],
            'shares': self.shares,
            'balance': self.balance,
            'return': (self.portfolio_values[-1] - self.initial_balance) / self.initial_balance
        }
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """
        Render environment state
        """
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"Price: {self.prices[self.current_step]:.2f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Shares: {self.shares:.6f}")
            print(f"Portfolio Value: {self.portfolio_values[-1]:.2f}")
            print(f"Return: {(self.portfolio_values[-1] - self.initial_balance) / self.initial_balance * 100:.2f}%")
        return None

class PPOTrader:
    """
    Wrapper class untuk menerapkan PPO dalam konteks trading
    """
    def __init__(self, prices, features=None, initial_investment=10000, lookback=60, forecast_days=30):
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
        lookback : int, optional
            Jumlah hari historis yang digunakan untuk prediksi, default 60
        forecast_days : int, optional
            Jumlah hari yang akan diprediksi ke depan, default 30
        """
        self.prices = np.array(prices)
        self.features = np.array(features) if features is not None else np.zeros((len(prices), 1))
        self.initial_investment = initial_investment
        
        # Hitung fitur dasar
        self.returns = np.zeros_like(prices)
        self.returns[1:] = (prices[1:] / prices[:-1]) - 1
        
        # Volatilitas (standar deviasi return rolling window)
        self.volatility = np.zeros_like(prices)
        window = 10
        for i in range(window, len(self.returns)):
            self.volatility[i] = np.std(self.returns[i-window+1:i+1])
        
        # State dimensi: [price, return, volatility, position, balance] + features
        self.state_dim = 5 + self.features.shape[1] + lookback
        self.action_dim = 3  # buy, sell, hold
        
        # Inisialisasi agent
        self.agent = PPOAgent(self.state_dim, self.action_dim)
        
        self.lookback = lookback
        self.forecast_days = forecast_days
        
        self.reset()
    
    def reset(self):
        """Reset state trading"""
        self.current_step = 0
        self.balance = self.initial_investment
        self.shares = 0
        self.position_value = 0
        self.portfolio_values = [self.initial_investment]
        return self._get_state()
    
    def _get_state(self):
        """Mendapatkan state saat ini"""
        price = self.prices[self.current_step]
        ret = self.returns[self.current_step]
        vol = self.volatility[self.current_step]
        position = self.shares * price / self.initial_investment
        balance = self.balance / self.initial_investment
        
        state = np.array([
            price / self.prices[0],  # Normalisasi harga
            ret,
            vol,
            position,
            balance
        ])
        
        # Tambahkan features jika ada
        if self.features is not None:
            state = np.concatenate([state, self.features[self.current_step]])
        
        # Tambahkan data historis
        if self.current_step >= self.lookback:
            state = np.concatenate([state, self.returns[self.current_step-self.lookback:self.current_step]])
        else:
            state = np.concatenate([state, np.zeros(self.lookback-self.current_step)])
        
        # Normalisasi data historis
        state[:self.lookback] = (state[:self.lookback] - state[:self.lookback].mean()) / (state[:self.lookback].std() + 1e-8)
        
        return state
    
    def step(self, action):
        """
        Melakukan langkah trading
        
        Parameters:
        -----------
        action : int
            0: buy, 1: sell, 2: hold
            
        Returns:
        --------
        tuple
            (next_state, reward, done, info)
        """
        current_price = self.prices[self.current_step]
        old_portfolio_value = self.balance + self.shares * current_price
        
        # Execute action
        if action == 0:  # Buy
            max_shares = self.balance / current_price
            self.shares += max_shares
            self.balance -= max_shares * current_price
        elif action == 1:  # Sell
            self.balance += self.shares * current_price
            self.shares = 0
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        if not done:
            new_price = self.prices[self.current_step]
            new_portfolio_value = self.balance + self.shares * new_price
            self.portfolio_values.append(new_portfolio_value)
            
            # Calculate reward (return)
            reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
            
            # Get new state
            next_state = self._get_state()
        else:
            reward = 0
            next_state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_values[-1],
            'shares': self.shares,
            'balance': self.balance
        }
        
        return next_state, reward, done, info
    
    def train(self, episodes=50, steps_per_episode=None):
        """
        Melatih agent PPO
        
        Parameters:
        -----------
        episodes : int
            Jumlah episode training
        steps_per_episode : int, optional
            Jumlah langkah per episode
            
        Returns:
        --------
        dict
            Hasil training
        """
        if steps_per_episode is None:
            steps_per_episode = len(self.prices) - 1
            
        best_reward = float('-inf')
        episode_rewards = []
        all_portfolio_values = []
        
        for episode in range(episodes):
            state = self.reset()
            states = []
            rewards = []
            done = False
            episode_reward = 0
            step = 0
            
            while not done and step < steps_per_episode:
                # Collect experience
                states.append(state)
                action, _ = self.agent.get_action(state)
                next_state, reward, done, _ = self.step(action)
                
                rewards.append(reward)
                episode_reward += reward
                state = next_state
                step += 1
            
            # Train agent with collected experience
            states = np.array(states)
            self.agent.train(states, n_epochs=10)
            
            episode_rewards.append(episode_reward)
            all_portfolio_values.append(self.portfolio_values)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
        
        return {
            'episode_rewards': episode_rewards,
            'portfolio_values': all_portfolio_values,
            'best_reward': best_reward
        }
    
    def predict(self, state):
        """
        Memprediksi aksi untuk state tertentu
        
        Parameters:
        -----------
        state : array-like
            State untuk diprediksi
            
        Returns:
        --------
        tuple
            (action, confidence)
        """
        return self.agent.get_action(state)
    
    def save(self, path):
        """Menyimpan model agent"""
        self.agent.save(path)
    
    def load(self, path):
        """Memuat model agent"""
        self.agent.load(path) 