#!/usr/bin/env python
"""
Cross-Ticker PPO Training with Optuna Hyperparameter Tuning
============================================================

This script trains a single PPO agent on multiple tickers simultaneously
to encourage generalization and prevent overfitting to any single asset.

Features:
- Cross-ticker training (10 assets: 5 US + 5 IDX)
- Percentage return normalization for cross-market compatibility
- Optuna hyperparameter optimization with MedianPruner
- Ticker embedding for asset-specific behavior

Author: AI Hedge Fund V2.3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.trading.ppo_agent import PPOTrader, PPOAgent, TradingEnv, VectorizedTradingEnv
from src.data.feature_engineering import TradingFeatureEngineer, prepare_macro_features


# Training universe (5 US + 5 IDX)
TRAINING_UNIVERSE = [
    # US Tech / Growth
    'NVDA', 'AAPL', 'MSFT', 'TSLA', 'GOOGL',
    # IDX Blue Chip
    'BBCA.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK', 'UNVR.JK'
]


def download_ticker_data(ticker, start_date, end_date):
    """Download and preprocess data for a single ticker."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, None
        
        # Flatten MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        prices = data['Close'].values.flatten()
        
        # Add macro features if available
        try:
            macro_data = prepare_macro_features(data)
            data = data.join(macro_data, how='left')
        except Exception:
            pass
        
        return prices, data
    except Exception as e:
        print(f"[Warning] Failed to download {ticker}: {e}")
        return None, None


def normalize_to_returns(prices):
    """
    Convert prices to percentage returns for cross-ticker normalization.
    This allows training on assets with vastly different price scales
    (e.g., NVDA ~$100 vs BBCA ~IDR 10,000).
    """
    returns = np.diff(prices) / prices[:-1]
    returns = np.clip(returns, -0.2, 0.2)  # Clip extreme moves
    # Prepend 0 to maintain length
    return np.concatenate([[0], returns])


class CrossTickerEnv:
    """
    Environment that samples from multiple tickers each episode.
    
    Each reset() selects a random ticker from the universe,
    promoting generalization across different asset characteristics.
    """
    
    def __init__(self, ticker_data_dict, **env_kwargs):
        """
        Args:
            ticker_data_dict: Dict mapping ticker -> (prices, ohlcv_df)
            **env_kwargs: Arguments passed to TradingEnv
        """
        self.ticker_data = ticker_data_dict
        self.tickers = list(ticker_data_dict.keys())
        self.env_kwargs = env_kwargs
        self.current_ticker = None
        self.current_env = None
        
        # Get observation space from first env
        first_ticker = self.tickers[0]
        prices, ohlcv = self.ticker_data[first_ticker]
        temp_env = TradingEnv(prices=prices, ohlcv_df=ohlcv, **env_kwargs)
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
    
    def reset(self):
        """Reset with a randomly selected ticker."""
        self.current_ticker = np.random.choice(self.tickers)
        prices, ohlcv = self.ticker_data[self.current_ticker]
        
        self.current_env = TradingEnv(
            prices=prices,
            ohlcv_df=ohlcv,
            **self.env_kwargs
        )
        return self.current_env.reset()
    
    def step(self, action):
        return self.current_env.step(action)
    
    def set_difficulty(self, fee, noise):
        if self.current_env and hasattr(self.current_env, 'set_difficulty'):
            self.current_env.set_difficulty(fee, noise)


def train_cross_ticker(
    tickers=TRAINING_UNIVERSE,
    episodes=200,
    start_date="2020-01-01",
    end_date=None,
    initial_investment=10000,
    verbose=True,
    **ppo_kwargs
):
    """
    Train a single PPO agent on multiple tickers for generalization.
    
    Returns:
        dict: Training results including Sharpe across tickers
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"[Cross-Ticker] Downloading data for {len(tickers)} tickers...")
    
    # Download all ticker data
    ticker_data = {}
    for ticker in tickers:
        prices, ohlcv = download_ticker_data(ticker, start_date, end_date)
        if prices is not None and len(prices) > 100:
            ticker_data[ticker] = (prices, ohlcv)
            if verbose:
                print(f"  ✓ {ticker}: {len(prices)} days")
        else:
            if verbose:
                print(f"  ✗ {ticker}: insufficient data")
    
    if len(ticker_data) < 2:
        raise ValueError("Need at least 2 tickers with valid data")
    
    print(f"[Cross-Ticker] Training on {len(ticker_data)} tickers...")
    
    # Create cross-ticker environment
    env = CrossTickerEnv(
        ticker_data,
        initial_balance=initial_investment,
        use_enhanced_features=True,
        transaction_fee=0.001
    )
    
    # Create PPO agent with correct state dimension
    state_dim = env.observation_space.shape[0]
    agent = PPOAgent(state_dim=state_dim, action_dim=3, **ppo_kwargs)
    
    # Training loop
    episode_rewards = []
    ticker_performance = {t: [] for t in ticker_data.keys()}
    
    for ep in range(episodes):
        state = env.reset()
        current_ticker = env.current_ticker
        
        total_reward = 0
        done = False
        
        # Curriculum learning
        progress = ep / episodes
        if progress < 0.33:
            env.set_difficulty(0.001, 0.0)
        elif progress < 0.66:
            env.set_difficulty(0.002, 0.01)
        else:
            env.set_difficulty(0.005, 0.02)
        
        while not done:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            total_reward += reward
        
        # Train after each episode
        if len(agent.buffer) > 0:
            agent.train()
        
        episode_rewards.append(total_reward)
        final_value = info.get('portfolio_value', initial_investment)
        ret = (final_value - initial_investment) / initial_investment
        ticker_performance[current_ticker].append(ret)
        
        if verbose and (ep + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {ep+1}/{episodes} | Ticker: {current_ticker} | "
                  f"Avg Reward: {avg_reward:.4f} | Return: {ret*100:.2f}%")
    
    # Calculate per-ticker Sharpe
    ticker_sharpes = {}
    for ticker, returns in ticker_performance.items():
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            ticker_sharpes[ticker] = sharpe
    
    avg_sharpe = np.mean(list(ticker_sharpes.values()))
    sharpe_std = np.std(list(ticker_sharpes.values()))
    
    print(f"\n[Cross-Ticker Results]")
    print(f"  Average Sharpe: {avg_sharpe:.3f} ± {sharpe_std:.3f}")
    print(f"  Ticker Sharpes: {ticker_sharpes}")
    
    return {
        'episode_rewards': episode_rewards,
        'ticker_performance': ticker_performance,
        'ticker_sharpes': ticker_sharpes,
        'avg_sharpe': avg_sharpe,
        'sharpe_variance': sharpe_std ** 2,
        'agent': agent
    }


def optuna_objective(trial, ticker_data, initial_investment=10000):
    """
    Optuna objective function for PPO hyperparameter tuning.
    
    Uses MedianPruner to stop unpromising trials early.
    """
    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    clip_ratio = trial.suggest_float('clip_ratio', 0.1, 0.3)
    entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    
    # Create environment
    env = CrossTickerEnv(
        ticker_data,
        initial_balance=initial_investment,
        use_enhanced_features=True,
        transaction_fee=0.001
    )
    
    # Create agent with trial hyperparameters
    state_dim = env.observation_space.shape[0]
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=3,
        lr=lr,
        gamma=gamma,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        batch_size=batch_size,
        lam=gae_lambda
    )
    
    # Training with pruning checkpoints
    n_epochs = 100
    epoch_sharpes = []
    
    for epoch in range(n_epochs):
        # Train for 10 episodes per epoch
        episode_returns = []
        for _ in range(10):
            state = env.reset()
            total_return = 0
            done = False
            
            while not done:
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, value, log_prob, done)
                state = next_state
            
            if len(agent.buffer) > 0:
                agent.train()
            
            final_value = info.get('portfolio_value', initial_investment)
            episode_returns.append((final_value - initial_investment) / initial_investment)
        
        # Calculate epoch Sharpe
        if len(episode_returns) > 1:
            epoch_sharpe = np.mean(episode_returns) / (np.std(episode_returns) + 1e-8) * np.sqrt(252)
        else:
            epoch_sharpe = 0
        
        epoch_sharpes.append(epoch_sharpe)
        
        # Report to Optuna for pruning
        trial.report(epoch_sharpe, epoch)
        
        # Prune if performance is poor
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Return average Sharpe from last 20 epochs
    return np.mean(epoch_sharpes[-20:])


def run_optuna_tuning(
    tickers=TRAINING_UNIVERSE[:5],  # Use subset for speed
    n_trials=50,
    start_date="2020-01-01",
    end_date=None,
    verbose=True
):
    """
    Run Optuna hyperparameter tuning for PPO.
    
    Returns:
        Best hyperparameters and study results
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"[Optuna] Downloading data for {len(tickers)} tickers...")
    
    # Download ticker data
    ticker_data = {}
    for ticker in tickers:
        prices, ohlcv = download_ticker_data(ticker, start_date, end_date)
        if prices is not None and len(prices) > 100:
            ticker_data[ticker] = (prices, ohlcv)
    
    if len(ticker_data) < 2:
        raise ValueError("Need at least 2 tickers with valid data")
    
    print(f"[Optuna] Starting hyperparameter optimization ({n_trials} trials)...")
    
    # Create study with MedianPruner
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: optuna_objective(trial, ticker_data),
        n_trials=n_trials,
        show_progress_bar=verbose
    )
    
    print(f"\n[Optuna Results]")
    print(f"  Best Sharpe: {study.best_value:.3f}")
    print(f"  Best Params: {study.best_params}")
    print(f"  Trials Completed: {len(study.trials)}")
    print(f"  Trials Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    return {
        'best_params': study.best_params,
        'best_sharpe': study.best_value,
        'study': study
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-Ticker PPO Training with Optuna")
    parser.add_argument('--mode', choices=['train', 'tune'], default='train',
                       help='Mode: train (cross-ticker) or tune (Optuna)')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of Optuna trials (tune mode)')
    parser.add_argument('--tickers', nargs='+', default=None,
                       help='Custom ticker list (default: TRAINING_UNIVERSE)')
    parser.add_argument('--start-date', default='2020-01-01',
                       help='Training data start date')
    parser.add_argument('--save-model', default=None,
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    tickers = args.tickers or TRAINING_UNIVERSE
    
    if args.mode == 'train':
        results = train_cross_ticker(
            tickers=tickers,
            episodes=args.episodes,
            start_date=args.start_date,
            verbose=True
        )
        
        if args.save_model:
            results['agent'].save_model(args.save_model)
        
        print(f"\n[Summary]")
        print(f"  Average Sharpe: {results['avg_sharpe']:.3f}")
        print(f"  Sharpe Variance: {results['sharpe_variance']:.4f}")
        
    elif args.mode == 'tune':
        results = run_optuna_tuning(
            tickers=tickers[:5],  # Use subset for speed
            n_trials=args.n_trials,
            start_date=args.start_date,
            verbose=True
        )
        
        print(f"\n[Best Configuration]")
        for key, value in results['best_params'].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
