"""
PPO Agent Module - PyTorch Version
===================================

Implementasi PPO (Proximal Policy Optimization) menggunakan PyTorch untuk trading.
Fitur: Entropy bonus, GAE, gradient clipping, LR scheduling, value clipping.

Enhanced Features (v2.0):
- Integrasi dengan TradingFeatureEngineer untuk signal scoring
- Support MACD, Stochastic RSI, Bollinger Bands, Volume scores
- ADX validation untuk filter sideways market
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces

# Import feature engineering (optional, for enhanced mode)
try:
    from src.data.feature_engineering import TradingFeatureEngineer, prepare_ppo_features
    HAS_FEATURE_ENGINEERING = True
except ImportError:
    HAS_FEATURE_ENGINEERING = False


class Buffer:
    """Buffer untuk menyimpan experience untuk PPO agent."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def __len__(self):
        return len(self.states)


class RunningMeanStd:
    """Dynamically calculates mean and std for feature normalization."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


import pandas as pd

class TradingEnv(gym.Env):
    """
    Trading environment yang kompatibel dengan gym untuk reinforcement learning.
    
    Enhanced Mode (use_enhanced_features=True):
        - 8 base portfolio features + 6 technical signal scores = 14 total features
        - Signal scores normalized 0.0-1.0 dengan 0.5 = neutral
        - ADX validation untuk filter sideways market
        
    Risk Management Features:
        - Position Sizing (Fixed Risk 2%)
        - Dynamic Stop Loss (ATR-based)
        - Trailing Stop
        - Regime Filter (EMA 200, ADX)
    """
    def __init__(self, prices, features=None, initial_balance=10000, 
                 transaction_fee=0.001, use_enhanced_features=False,
                 ohlcv_df=None, observation_noise=0.0, macro_features=None,
                 slippage_range=(0.0, 0.0005)):
        super(TradingEnv, self).__init__()

        self.prices = np.array(prices).flatten()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.use_enhanced_features = use_enhanced_features
        self.ohlcv_df = ohlcv_df
        self.observation_noise = observation_noise
        
        # Global Macro Features (NASDAQ, DJI, TNX, VIX log returns)
        # Global Macro Features (NASDAQ, DJI, TNX, VIX log returns)
        self.macro_features = macro_features  # Shape: (n_steps, 4)
        self.n_macro_features = 4 if macro_features is not None else 0
        
        # Dynamic Scaling for Macro Features
        if self.n_macro_features > 0:
            self.macro_ms = RunningMeanStd(shape=(self.n_macro_features,))
            # Init with full history for stable scaling
            self.macro_ms.update(self.macro_features)
        else:
            self.macro_ms = None
        
        # Slippage Simulation (Anti-Overfit)
        self.slippage_range = slippage_range
        
        
        # Risk Management Config
        self.risk_per_trade = 0.02  # 2% Risk per trade
        self.atr_multiplier = 2.0   # Stop Loss distance (2x ATR)
        
        # Pre-calculate Indicators for Risk Management
        prices_series = pd.Series(self.prices)
        
        # 1. EMA 200 (Trend Filter)
        self.ema200 = prices_series.ewm(span=200, adjust=False).mean().values
        
        # 2. ATR (Volatility for Stop Loss)
        if ohlcv_df is not None:
            # Use True Range from OHLCV
            high = ohlcv_df['High']
            low = ohlcv_df['Low']
            close = ohlcv_df['Close']
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.atr = tr.rolling(14).mean().bfill().values
            
            # ADX for Regime Filter
            # (Assumes TradingFeatureEngineer calculates it, or we calculate approximations)
            # For simplicity, we'll rely on ATR and EMA here, or use feature_engineer output if available.
        else:
            # Approx ATR from Close only
            tr = prices_series.diff().abs()
            self.atr = tr.rolling(14).mean().bfill().values
        
        # Enhanced features mode
        if use_enhanced_features and HAS_FEATURE_ENGINEERING:
            self.feature_engineer = TradingFeatureEngineer(
                ohlcv_df=ohlcv_df, 
                prices=self.prices
            )
            self.enhanced_features = self.feature_engineer.get_features_for_ppo_all()
            # 8 base + 7 enhanced = 15 features (added Weekly Trend)
            self.n_enhanced_features = 7
        else:
            self.feature_engineer = None
            self.enhanced_features = None
            self.n_enhanced_features = 0
        
        # External features (for backward compatibility)
        self.features = features if features is not None else np.zeros((len(prices), 1))

        # State shape: base (8) + enhanced (6) + macro (3) + external features
        if use_enhanced_features and self.enhanced_features is not None:
            obs_dim = 8 + self.n_enhanced_features + self.n_macro_features
        else:
            obs_dim = 8 + self.features.shape[1] + self.n_macro_features
            
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        self.done = False
        self.portfolio_value_history = [self.initial_balance]
        self.avg_entry_price = 0
        self.position_value = 0
        self.unrealized_pnl = 0
        
        # Risk Management State
        self.stop_loss_price = 0.0
        self.peak_price_since_entry = 0.0
        self.prev_drawdown = 0.0
        
        # Differential Sharpe Ratio State
        self.dsr_m1 = 0.0 # Running mean return
        self.dsr_m2 = 0.0 # Running mean square return

        return self._get_observation()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, {}

        current_price = self.prices[self.current_step]
        shares_before_action = self.shares
        prev_portfolio_value = self.balance + self.shares * current_price
        
        # ---------------------------------------------------------
        # 1. RISK MANAGEMENT CHECKS (Before Action)
        # ---------------------------------------------------------
        forced_sell = False
        current_atr = self.atr[self.current_step] if hasattr(self, 'atr') else current_price * 0.02
        
        if self.shares > 0:
            # Dynamic Stop Loss / Trailing Stop Logic
            # Check if Hit Stop Loss
            if current_price < self.stop_loss_price:
                action = 2  # Force Sell (Stop Loss Hit)
                forced_sell = True
            else:
                # Trailing Stop: Raise SL if price moves up
                # Logic: Maintain SL at (Price - 2*ATR) distance if it raises the SL level
                potential_new_sl = current_price - (current_atr * self.atr_multiplier)
                if potential_new_sl > self.stop_loss_price:
                    self.stop_loss_price = potential_new_sl

        # ---------------------------------------------------------
        # 2. REGIME FILTER (Trend Filter) - RESTORED FOR ROBUSTNESS
        # ---------------------------------------------------------
        # Filter Buy (Action 1) if Price < EMA 200 (Downtrend)
        is_uptrend = current_price > self.ema200[self.current_step]
        if action == 1 and not is_uptrend:
            action = 0  # Force Hold/Skip Buy

        # ---------------------------------------------------------
        # 3. ACTION EXECUTION
        # ---------------------------------------------------------
        if action == 1:  # Buy
            if self.balance > 0:
                # Position Sizing (Fixed Risk 1-2%)
                risk_amount = self.balance * self.risk_per_trade  # e.g., 2% of balance
                stop_loss_dist = current_atr * self.atr_multiplier
                
                if stop_loss_dist > 0:
                    shares_to_buy = risk_amount / stop_loss_dist
                else:
                    shares_to_buy = 0
                
                # Cap at available cash (cannot borrow)
                max_shares_balance = self.balance / (current_price * (1 + self.transaction_fee))
                shares_to_buy = min(shares_to_buy, max_shares_balance)
                
                if shares_to_buy > 0:
                    # Slippage Simulation (Volatility Aware)
                    # Base slippage + Volatility penalty
                    # Example: 0.0001 + (ATR/Price * 0.1)
                    if hasattr(self, 'atr'):
                        vol_ratio = current_atr / current_price
                        slippage = self.slippage_range[0] + vol_ratio * 0.1
                        slippage = min(slippage, self.slippage_range[1]) # Cap at max slippage
                    else:
                         slippage = np.random.uniform(self.slippage_range[0], self.slippage_range[1])
                    
                    execution_price = current_price * (1 + slippage)
                    
                    cost = shares_to_buy * execution_price * (1 + self.transaction_fee)
                    
                    if self.shares > 0:
                        # Averaging Up/Down
                        self.avg_entry_price = (self.shares * self.avg_entry_price + shares_to_buy * execution_price) / (self.shares + shares_to_buy)
                        # On scaling in, maybe adjust SL? Keep existing SL or move it up?
                        # For safety, let's keep the tighter of existing vs new calculation
                        new_sl = current_price - stop_loss_dist
                        self.stop_loss_price = max(self.stop_loss_price, new_sl)
                    else:
                        # New Position
                        self.avg_entry_price = execution_price
                        self.stop_loss_price = current_price - stop_loss_dist

                    self.shares += shares_to_buy
                    self.balance -= cost

        elif action == 2:  # Sell
            if self.shares > 0:
                # Slippage Simulation (Sell at worse price) (Volatility Aware)
                if hasattr(self, 'atr'):
                    vol_ratio = current_atr / current_price
                    slippage = self.slippage_range[0] + vol_ratio * 0.1
                    slippage = min(slippage, self.slippage_range[1])
                else:
                    slippage = np.random.uniform(self.slippage_range[0], self.slippage_range[1])
                    
                execution_price = current_price * (1 - slippage)
                
                sell_value = self.shares * execution_price * (1 - self.transaction_fee)
                self.balance += sell_value
                self.shares = 0
                self.avg_entry_price = 0
                self.stop_loss_price = 0.0 # Reset SL

        # Move to next step
        self.current_step += 1

        if self.current_step >= len(self.prices) - 1:
            self.done = True

        # Calculate new portfolio value
        new_price = self.prices[self.current_step]
        self.position_value = self.shares * new_price
        new_portfolio_value = self.balance + self.position_value

        # Update unrealized P&L
        if self.shares > 0 and self.avg_entry_price > 0:
            self.unrealized_pnl = (new_price - self.avg_entry_price) * self.shares
        else:
            self.unrealized_pnl = 0

        self.portfolio_value_history.append(new_portfolio_value)

        # Calculate Drawdown from Peak
        max_portfolio = max(self.portfolio_value_history)
        drawdown = (max_portfolio - new_portfolio_value) / max_portfolio if max_portfolio > 0 else 0

        # ---------------------------------------------------------
        # 4. REWARD SHAPING: Differential Sharpe Ratio (DSR)
        # ---------------------------------------------------------
        # Calculate current step return
        current_return = (new_portfolio_value / prev_portfolio_value) - 1
        
        # Update running statistics (Exponential Moving Average)
        eta = 0.01  # Adaptation rate (approx 100-step memory)
        delta_m1 = current_return - self.dsr_m1
        self.dsr_m1 += eta * delta_m1
        self.dsr_m2 += eta * (current_return**2 - self.dsr_m2)
        
        # Calculate DSR
        # D_t = (B_{t-1} * (R_t - A_{t-1}) - 0.5 * A_{t-1} * (R_t^2 - B_{t-1})) / (B_{t-1} - A_{t-1}^2)^(3/2)
        # Simplified:
        denominator = np.sqrt(self.dsr_m2 - self.dsr_m1**2 + 1e-9)
        # Prevent division by zero or explosive rewards during initialization
        if denominator < 1e-6:
             dsr_reward = current_return * 10 # Fallback to simple return
        else:
             dsr_reward = (self.dsr_m2 * delta_m1 - 0.5 * self.dsr_m1 * (current_return**2 - self.dsr_m2)) / (denominator**3 + 1e-9)
        
        # Scale DSR to be meaningful (DSR is typically small)
        reward = np.clip(dsr_reward, -5.0, 5.0)
        
        # Explicit Transaction Penalty (Still needed to discourage churning)
        if action != 0 and self.shares != shares_before_action: # Only if actually traded
             reward -= self.transaction_fee * 5
             
        # Stop Loss Penalty (Nudge)
        if forced_sell:
             reward -= 0.5
             
        # Drawdown Penalty (Secondary objective)
        delta_drawdown = max(0, drawdown - self.prev_drawdown)
        reward -= (delta_drawdown * 5)
            
        # Activity Reward (Motivation to Trade)
        # Bonus for taking action AND being profitable
        ratio = new_portfolio_value / prev_portfolio_value
        if action != 0 and ratio > 1.0:
            reward += 0.05 
            
        # B. DIFFERENTIAL DRAWDOWN PENALTY (Optional, kept small)
        # Keeping it but relying mostly on Log-Utility
        # delta_drawdown = max(0, drawdown - self.prev_drawdown)
        # reward -= (delta_drawdown * 10) 
        
        # Update prev_drawdown
        self.prev_drawdown = drawdown

        # C. ACTION TAX REMOVED
        # if action != 0: reward -= 0.001 (Removed)
            
        # Additional penalty for hitting Stop Loss? (Optional, implied by loss)
        # Additional penalty for hitting Stop Loss? (Optional, implied by loss)
        # if forced_sell:
        #    reward -= 1.0 # Removed (Too harsh)

        # End episode bonus
        if self.done:
            final_return = (new_portfolio_value - self.initial_balance) / self.initial_balance
            if len(self.portfolio_value_history) > 1:
                returns = np.diff(self.portfolio_value_history) / np.array(self.portfolio_value_history[:-1])
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                # Significant bonus for high Sharpe (Stability)
                reward += final_return + 0.5 * max(sharpe, 0)

        return self._get_observation(), reward, self.done, {
            'portfolio_value': new_portfolio_value,
            'balance': self.balance,
            'shares': self.shares,
            'unrealized_pnl': self.unrealized_pnl,
            'position_value': self.position_value,
            'stop_loss_hit': forced_sell
        }

    def _get_observation(self):
        current_price = self.prices[self.current_step]

        # Technical indicators (base)
        if self.current_step >= 20:
            sma_short = np.mean(self.prices[self.current_step-5:self.current_step+1])
            sma_long = np.mean(self.prices[self.current_step-20:self.current_step+1])
            price_sma_ratio_short = current_price / sma_short - 1
            price_sma_ratio_long = current_price / sma_long - 1

            price_window = self.prices[self.current_step-20:self.current_step+1]
            returns = np.diff(price_window) / price_window[:-1]
            volatility = np.std(returns)
        else:
            price_sma_ratio_short = 0
            price_sma_ratio_long = 0
            volatility = 0.02

        # Calculate Log Return (Relative Feature)
        if self.current_step > 0:
            log_return = np.log(current_price / self.prices[self.current_step - 1])
        else:
            log_return = 0.0

        # Base portfolio info (8 features)
        base_info = np.array([
            self.balance / self.initial_balance,
            self.shares * current_price / self.initial_balance,
            log_return,  # Replaced absolute price ratio
            self.position_value / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            price_sma_ratio_short,
            price_sma_ratio_long,
            volatility * 10
        ], dtype=np.float32)

        # Enhanced mode: use pre-computed signal scores
        if self.use_enhanced_features and self.enhanced_features is not None:
            # Get pre-computed features for current step (6 features)
            tech_features = self.enhanced_features[self.current_step]
            obs = np.concatenate([base_info, tech_features])
        else:
            # Legacy mode: use external features
            try:
                current_features = self.features[self.current_step]
                if not isinstance(current_features, np.ndarray):
                    current_features = np.array(current_features, dtype=np.float32)
                current_features = current_features.flatten().astype(np.float32)
            except (IndexError, TypeError):
                current_features = np.zeros(self.features.shape[1], dtype=np.float32)
            obs = np.concatenate([base_info, current_features])
        
        # Add Macro Features (NASDAQ, DJI, TNX log returns)
        if self.macro_features is not None and self.current_step < len(self.macro_features):
            macro = self.macro_features[self.current_step]
            if not isinstance(macro, np.ndarray):
                macro = np.array(macro, dtype=np.float32)
            # Dynamic Scaling using RunningMeanStd
            # Update scaler (optional: only update during training? For now, update always to adapt)
            # self.macro_ms.update(macro) # Might cause drift during inference if not careful.
            # Ideally, freeze stats during inference. But let's keep it simple: Use current stats.
            
            macro_norm = (macro - self.macro_ms.mean) / (np.sqrt(self.macro_ms.var) + 1e-8)
            macro_scaled = np.clip(macro_norm, -5.0, 5.0) # Clip extreme outliers
            obs = np.concatenate([obs, macro_scaled.flatten()])
        elif self.n_macro_features > 0:
            # Pad with zeros if macro data unavailable
            obs = np.concatenate([obs, np.zeros(self.n_macro_features, dtype=np.float32)])
        
        # Add Noise Injection (Training only)
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, obs.shape)
            obs += noise
            
        return obs.astype(np.float32)

    def set_difficulty(self, fee, noise):
        """Update difficulty parameters for curriculum learning."""
        self.transaction_fee = fee
        self.observation_noise = noise


class ActorCritic(nn.Module):
    """
    Actor-Critic network untuk PPO menggunakan PyTorch.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        shared_out = self.shared(state)
        action_probs = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_probs, value
    
    def get_action(self, state):
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)
    
    def evaluate(self, states, actions):
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    """
    PPO Agent menggunakan PyTorch dengan entropy bonus dan gradient clipping.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        clip_ratio=0.2,
        batch_size=64,
        epochs=10,
        lam=0.95,
        entropy_coef=0.005,
        value_coef=0.5,
        max_grad_norm=0.5,
        device=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        self.buffer = Buffer()
        self.target_kl = 0.015

    def get_action(self, state):
        """Select action based on current state."""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        if len(state.shape) == 1:
            if state.shape[0] != self.state_dim:
                if state.shape[0] > self.state_dim:
                    state = state[:self.state_dim]
                else:
                    state = np.pad(state, (0, self.state_dim - state.shape[0]))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor)
        
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in buffer."""
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.values.append(value)
        self.buffer.log_probs.append(log_prob)
        self.buffer.dones.append(done)

    def calculate_advantages(self):
        """Calculate advantages using GAE."""
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        next_value = 0
        next_advantage = 0

        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]
            returns[t] = rewards[t] + self.gamma * next_value * mask
            td_error = rewards[t] + self.gamma * next_value * mask - values[t]
            advantages[t] = td_error + self.gamma * self.lam * next_advantage * mask

            next_value = values[t]
            next_advantage = advantages[t]

        return returns, advantages

    def train(self, batch_size=None):
        """Train PPO agent."""
        if len(self.buffer) == 0:
            return {}

        batch_size = batch_size or self.batch_size
        
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)

        returns, advantages = self.calculate_advantages()
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.epochs):
            indices = torch.randperm(len(states))

            for start in range(0, len(indices), batch_size):
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate current policy
                log_probs, values, entropy = self.network.evaluate(batch_states, batch_actions)

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                
                # Critic loss
                critic_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

            # KL divergence check
            with torch.no_grad():
                new_log_probs, _, _ = self.network.evaluate(states, actions)
                kl_div = (old_log_probs - new_log_probs).mean().item()
                
            if abs(kl_div) > 1.5 * self.target_kl:
                break

        self.scheduler.step()
        self.buffer.clear()
        
        return {
            'actor_loss': total_actor_loss / max(n_updates, 1),
            'critic_loss': total_critic_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1)
        }

    def save_model(self, path: str):
        """
        Save PPO network weights to disk.
        
        Args:
            path: File path to save the model (e.g., 'saved_models/ppo/NVDA.pt')
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)
        print(f"[PPO] Model saved to {path}")

    def load_model(self, path: str):
        """
        Load PPO network weights from disk.
        
        Args:
            path: File path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"[PPO] Model loaded from {path}")


class PPOTrader:
    """
    Wrapper class untuk menerapkan PPO dalam konteks trading.
    
    Args:
        prices: Array harga Close
        features: Optional external features (legacy mode)
        initial_investment: Modal awal
        log_dir: Direktori untuk logging
        use_enhanced_features: Jika True, gunakan signal scoring dari TradingFeatureEngineer
        ohlcv_df: DataFrame OHLCV untuk enhanced features (opsional, lebih akurat)
    """
    def __init__(self, prices, features=None, initial_investment=10000, log_dir=None,
                 use_enhanced_features=False, ohlcv_df=None, transaction_fee=0.001, 
                 train_noise_level=0.0, macro_features=None, slippage_range=(0.0, 0.0005)):
        
        self.prices = np.array(prices).flatten()
        self.initial_investment = initial_investment
        self.use_enhanced_features = use_enhanced_features
        self.ohlcv_df = ohlcv_df
        self.transaction_fee = transaction_fee
        self.train_noise_level = train_noise_level
        self.macro_features = macro_features
        self.slippage_range = slippage_range
        
        # Enhanced mode: use signal scoring
        if use_enhanced_features and HAS_FEATURE_ENGINEERING:
            self.features = None  # Not used in enhanced mode
            self.env = TradingEnv(
                self.prices, 
                features=None,
                initial_balance=initial_investment,
                use_enhanced_features=True,
                ohlcv_df=ohlcv_df,
                transaction_fee=transaction_fee,
                observation_noise=train_noise_level,
                macro_features=macro_features,
                slippage_range=slippage_range
            )
        else:
            # Legacy mode: generate basic features if not provided
            if features is None:
                prices_arr = self.prices
                returns = np.zeros_like(prices_arr)
                returns[1:] = (prices_arr[1:] / prices_arr[:-1]) - 1

                volatility = np.zeros_like(prices_arr)
                window = 10
                for i in range(window, len(returns)):
                    volatility[i] = np.std(returns[i-window+1:i+1])

                features = np.column_stack((returns, volatility))
            
            self.features = features
            self.env = TradingEnv(self.prices, features, initial_investment)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.agent = PPOAgent(
            state_dim, 
            action_dim,
            entropy_coef=0.05,  # Increased from 0.02 to encourage exploration
            epochs=8,
            lr=3e-4  # Explicit LR
        )
        self.trained = False
        self.log_dir = log_dir

    def train(self, episodes=100, max_steps=None, verbose=True):
        """Train PPO agent."""
        if max_steps is None:
            max_steps = len(self.prices) - 1

        episode_rewards = []
        portfolio_values = []
        best_reward = -np.inf

        for ep in range(episodes):
            # Curriculum Learning Logic
            # Phase 1 (0-33%): Easy (Fee 0.1%, Noise 0)
            # Phase 2 (33-66%): Medium (Fee 0.2%, Noise 0.01)
            # Phase 3 (66-100%): Hard (Fee 0.5%, Noise 0.02)
            if ep < episodes * 0.33:
                curr_fee = 0.001
                curr_noise = 0.0
                phase = "Easy"
            elif ep < episodes * 0.66:
                curr_fee = 0.002
                curr_noise = 0.01
                phase = "Medium"
            else:
                curr_fee = 0.005 # Increased fee pressure
                curr_noise = 0.02 # Chaos mode
                phase = "Hard"
                
            if hasattr(self.env, 'set_difficulty'):
                self.env.set_difficulty(curr_fee, curr_noise)
            
            # Entropy Decay (Learning stabilization)
            # Start 0.05 (Exploration) -> End 0.001 (Exploitation)
            start_entropy = 0.05
            end_entropy = 0.001
            decay_rate = 0.99
            
            # Linear Decay based on progress
            progress = ep / episodes
            current_entropy = start_entropy * (1 - progress) + end_entropy * progress
            current_entropy = max(end_entropy, current_entropy)
            
            # Update agent entropy
            if hasattr(self.agent, 'entropy_coef'):
                self.agent.entropy_coef = current_entropy
                
            state = self.env.reset()
            total_reward = 0
            
            for t in range(max_steps):
                action, log_prob, value = self.agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)

                self.agent.store_transition(state, action, reward, value, log_prob, done)

                state = next_state
                total_reward += reward

                if done:
                    break

            # Train after each episode
            train_info = self.agent.train()

            episode_rewards.append(total_reward)
            final_value = info.get('portfolio_value', self.initial_investment)
            portfolio_values.append(final_value)

            if total_reward > best_reward:
                best_reward = total_reward

            if verbose and (ep + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_value = np.mean(portfolio_values[-10:])
                print(f"Episode {ep+1}/{episodes} [{phase}] | Avg Reward: {avg_reward:.4f} | "
                      f"Avg Portfolio: {avg_value:.2f} | Best: {best_reward:.4f}")

        self.trained = True

        return {
            'episode_rewards': episode_rewards,
            'portfolio_values': portfolio_values,
            'best_reward': best_reward
        }

    def backtest(self, prices=None, features=None, initial_investment=None):
        """Run backtest dengan trained agent."""
        if prices is not None or features is not None or initial_investment is not None:
            test_prices = prices if prices is not None else self.prices
            test_features = features if features is not None else self.features
            test_initial = initial_investment if initial_investment is not None else self.initial_investment
            test_env = TradingEnv(
                test_prices, 
                test_features, 
                test_initial,
                transaction_fee=self.transaction_fee,
                use_enhanced_features=self.use_enhanced_features,
                ohlcv_df=self.ohlcv_df
            )
        else:
            test_env = self.env

        state = test_env.reset()
        done = False
        trades = []
        actions_taken = []

        while not done:
            action, _, _ = self.agent.get_action(state)
            actions_taken.append(action)

            current_price = test_env.prices[test_env.current_step]
            current_shares = test_env.shares
            current_step = test_env.current_step

            next_state, reward, done, info = test_env.step(action)

            # Record trades
            if action == 1 and info['shares'] > current_shares:
                trades.append({
                    'day': current_step,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': info['shares'] - current_shares,
                    'value': (info['shares'] - current_shares) * current_price
                })
            elif action == 2 and info['shares'] < current_shares:
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

        # Max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, dd)

        # Sharpe ratio
        if len(portfolio_values) > 1:
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Win rate
        win_trades = sum(1 for i in range(0, len(trades)-1, 2) 
                        if i+1 < len(trades) and trades[i+1]['value'] > trades[i]['value'])
        total_pairs = len(trades) // 2
        win_rate = (win_trades / total_pairs * 100) if total_pairs > 0 else 0

        performance = {
            'initial_investment': test_env.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
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
    
    def save(self, path):
        """Save model checkpoint."""
        self.agent.save_model(path)
        self.trained = True
    
    def load(self, path):
        """Load model checkpoint."""
        self.agent.load_model(path)
        self.trained = True


class MultiAssetTradingEnv(gym.Env):
    """Stub untuk multi-asset trading environment."""
    def __init__(self, prices_dict, features_dict=None, initial_balance=10000, transaction_fee=0.001):
        super().__init__()
        self.prices_dict = prices_dict
        self.features_dict = features_dict or {k: None for k in prices_dict}
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee

    def reset(self):
        pass

    def step(self, action):
        pass

    def _get_observation(self):
        pass