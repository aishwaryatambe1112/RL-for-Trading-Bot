import os
import random
import math
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

CSV_PATH = r"D:\DQN\AMZN.csv"  

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment / data
OBS_WINDOW = 10         # number of past timesteps used for state
FEATURES = ["Open", "High", "Low", "Close", "Volume"]  # expected CSV columns

# Training hyperparams
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
REPLAY_BUFFER_SIZE = 5000
MIN_REPLAY_SIZE = 300
TARGET_UPDATE_FREQ = 100  # in steps
MAX_EPISODES = 200        # number of passes through data (episodes)
MAX_STEPS_PER_EPISODE = None  # use full dataset length - OBS_WINDOW by default
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY_STEPS = 500
TRAINING_FREQ = 1         # update every step
SAVE_MODEL_EVERY = 10     # epochs
TRANSACTION_COST = 0.001  # 0.1% per trade

MODEL_SAVE_PATH = "dqn_amzn.pth"

# ------------------------
# Utility / Replay Buffer
# ------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ------------------------
# Neural Network (Q)
# ------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------
# Trading Environment
# ------------------------
class TradingEnv:
    def __init__(self, df: pd.DataFrame, obs_window: int = OBS_WINDOW, transaction_cost: float = TRANSACTION_COST):
        self.df = df.reset_index(drop=True)
        self.obs_window = obs_window
        self.transaction_cost = transaction_cost
        self._max_index = len(df) - 1
        self.reset()

    def reset(self, start_index: int = None):
        # choose start index such that we have obs_window history and at least one step forward
        if start_index is None:
            start_index = self.obs_window
        self.idx = start_index
        self.position = 0  # 0 = flat, 1 = long (holding one share)
        self.cash = 1.0    # normalized portfolio value; we start with 1.0 unit cash
        self.shares = 0.0
        self.portfolio_value = 1.0
        self.done = False
        return self._get_state()

    def _get_price(self, index):
        return float(self.df.loc[index, "Close"])

    def _get_state(self):
        # gather past obs_window rows of OHLCV and flatten
        start = self.idx - self.obs_window
        window = self.df.loc[start:self.idx - 1, FEATURES].values  # shape (obs_window, n_features)
        # if insufficient rows, pad (shouldn't happen with initialization)
        if window.shape[0] < self.obs_window:
            pad = np.zeros((self.obs_window - window.shape[0], window.shape[1]))
            window = np.vstack([pad, window])
        # normalize by previous close (scale-invariant)
        # we'll pass pre-scaled features externally: see data pre-processing code
        flat = window.flatten()
        extras = np.array([self.position, self.cash, self.portfolio_value])
        return np.concatenate([flat, extras]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: 0 hold, 1 buy, 2 sell
        """
        if self.done:
            raise RuntimeError("Step called on terminated environment. Call reset().")

        prev_price = self._get_price(self.idx - 1)
        cur_price = self._get_price(self.idx)

        info = {}

        # Execute action
        executed = False
        trade_cost = 0.0

        if action == 1:  # Buy
            if self.position == 0:
                # buy one share with available cash at current price
                # We normalize by starting portfolio = 1.0; buy uses fraction of portfolio equal to price / portfolio_value
                self.shares = (self.cash / cur_price)  # buy as many fractionally as cash allows (allow fractional shares)
                trade_cost = cur_price * self.shares * self.transaction_cost
                self.cash -= (cur_price * self.shares + trade_cost)
                self.position = 1
                executed = True
        elif action == 2:  # Sell
            if self.position == 1:
                # sell all shares at current price
                proceeds = cur_price * self.shares
                trade_cost = proceeds * self.transaction_cost
                self.cash += (proceeds - trade_cost)
                self.shares = 0.0
                self.position = 0
                executed = True
        # else hold -> nothing

        # compute new portfolio value (cash + holdings)
        holdings_value = cur_price * self.shares
        prev_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + holdings_value

        # reward: change in portfolio value (could be scaled)
        reward = (self.portfolio_value - prev_portfolio_value)

        # step forward
        self.idx += 1
        if self.idx > self._max_index:
            self.done = True
        else:
            self.done = False

        next_state = self._get_state() if not self.done else np.zeros_like(self._get_state())

        # info
        info['executed'] = executed
        info['trade_cost'] = trade_cost
        info['portfolio_value'] = self.portfolio_value
        info['cash'] = self.cash
        info['shares'] = self.shares
        info['position'] = self.position

        return next_state, float(reward), self.done, info

# ------------------------
# Agent
# ------------------------
class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = QNetwork(state_dim, n_actions).to(DEVICE)
        self.target_net = QNetwork(state_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.total_steps = 0
        self.epsilon = EPS_START

    def select_action(self, state: np.ndarray, explore=True):
        # epsilon-greedy
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                qvals = self.policy_net(state_v)
            return int(torch.argmax(qvals).item())

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_epsilon(self):
        # linear decay
        self.total_steps += 1
        fraction = min(float(self.total_steps) / EPS_DECAY_STEPS, 1.0)
        self.epsilon = EPS_START + fraction * (EPS_END - EPS_START)

    def train_step(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(np.vstack(transitions.state), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(transitions.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(np.vstack(transitions.next_state), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(transitions.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # Double DQN approach: use policy_net to select argmax, target_net to evaluate
        with torch.no_grad():
            next_q_vals_policy = self.policy_net(next_states)
            next_actions = torch.argmax(next_q_vals_policy, dim=1, keepdim=True)
            next_q_vals_target = self.target_net(next_states)
            next_q_value = next_q_vals_target.gather(1, next_actions)

            expected_q = rewards + (1 - dones) * self.gamma * next_q_value

        loss = self.loss_fn(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path=MODEL_SAVE_PATH):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# ------------------------
# Data loading & preprocessing
# ------------------------
def load_and_preprocess(csv_path: str, features=FEATURES, obs_window=OBS_WINDOW):
    df = pd.read_csv(csv_path)
    # Ensure needed columns exist
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    # Sort by date if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    # Keep only required columns
    df = df[features].copy()

    # Create returns and normalized log price features optionally
    # We'll use raw OHLCV scaled via StandardScaler applied rolling window wise.
    scaler = StandardScaler()
    # For stable training, fit a scaler on the full matrix (could be done on training split)
    scaled = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled, columns=features)

    # Merge back into df_scaled
    df_scaled = df_scaled.rename(columns={c: f"{c}_s" for c in features})

    # Create final dataframe with scaled columns, keep original close for price reference as well
    df_out = pd.concat([df, df_scaled], axis=1)
    return df_out

# ------------------------
# Main training loop
# ------------------------
def train():
    print("Loading data...", CSV_PATH)
    df = load_and_preprocess(CSV_PATH, FEATURES, OBS_WINDOW)

    # Build dataset where state vector consists of last OBS_WINDOW scaled features flattened
    # We will create a dataframe with scaled columns names like Close_s etc.
    scaled_cols = [f"{c}_s" for c in FEATURES]

    # Prepare env-ready df containing scaled features and original close for price queries
    df_for_env = df.copy()
    # The env expects columns named in FEATURES. We'll temporarily override those with scaled names for the window.
    for i, c in enumerate(FEATURES):
        df_for_env[c] = df[f"{c}_s"]

    env = TradingEnv(df_for_env, obs_window=OBS_WINDOW, transaction_cost=TRANSACTION_COST)

    # state dim calculation: obs_window * n_features + extras (position, cash, portfolio_value)
    state_dim = OBS_WINDOW * len(FEATURES) + 3
    n_actions = 3

    agent = DQNAgent(state_dim=state_dim, n_actions=n_actions, lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE)

    total_steps = 0
    losses = []
    best_portfolio = -np.inf

    # We'll run multiple episodes. Each episode will start at a random index in dataset (so agent sees different slices).
    max_start = len(df_for_env) - 2 - OBS_WINDOW
    if max_start <= OBS_WINDOW:
        raise ValueError("CSV too short for chosen OBS_WINDOW.")

    for ep in range(1, MAX_EPISODES + 1):
        # randomize start so episodes see different segments
        start_idx = random.randint(OBS_WINDOW, max_start)
        state = env.reset(start_index=start_idx)
        done = False
        ep_reward = 0.0
        steps = 0

        # If a per-episode step limit desired:
        while not done:
            # select action with current policy (explore)
            action = agent.select_action(state, explore=True)

            next_state, reward, done, info = env.step(action)
            ep_reward += reward

            agent.push_transition(state, action, reward, next_state, done)

            # Epsilon decay and training step
            agent.update_epsilon()

            if total_steps % TRAINING_FREQ == 0:
                loss_val = agent.train_step()
                if loss_val is not None:
                    losses.append(loss_val)

            # update target network periodically
            if total_steps % TARGET_UPDATE_FREQ == 0 and total_steps > 0:
                agent.update_target()

            state = next_state
            steps += 1
            total_steps += 1

            # safety break if too many steps
            if MAX_STEPS_PER_EPISODE and steps >= MAX_STEPS_PER_EPISODE:
                break

        final_value = env.portfolio_value
        if final_value > best_portfolio:
            best_portfolio = final_value
            agent.save("best_" + MODEL_SAVE_PATH)

        print(f"Episode {ep:03d} | StartIdx {start_idx} | Steps {steps} | EpReward {ep_reward:.5f} | "
              f"Final PV {final_value:.5f} | Eps {agent.epsilon:.4f} | Replay {len(agent.replay_buffer)}")

        if ep % SAVE_MODEL_EVERY == 0:
            agent.save(MODEL_SAVE_PATH)

    # final save
    agent.save(MODEL_SAVE_PATH)
    print("Training finished. Best portfolio value:", best_portfolio)
    print("Model saved to", MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()
