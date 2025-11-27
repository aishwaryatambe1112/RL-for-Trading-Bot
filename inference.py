# inference.py
"""
Refined inference script for your DQN trading agent.

Features:
 - Keeps Close_orig (unscaled) for trade math, scaled features for agent state.
 - Two trade modes: single-share (default) or fractional (use --fractional).
 - Realistic starting cash via --starting-cash (default 10000.0).
 - Logs Q-values per step to qvals_log.csv.
 - Saves per-step CSV (inference_trades.csv) and improved plot (inference_trades_plot_improved.png).
 - Robust checkpoint lookup (or pass --checkpoint).

Usage examples:
 python inference.py
 python inference.py --fractional
 python inference.py --starting-cash 5000 --checkpoint best_amzn.pth --csv AMZN.csv
"""

import os
import argparse
from collections import namedtuple
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------- Config defaults ----------------
DEFAULT_CSV = "AMZN.csv"
CHECKPOINT_CANDIDATES = [
    "best_dqn_amzn.pth",
    "best_amzn.pth",
    "best_amaz.pth",
    "dqn_amzn.pth",
    "checkpoint.pth"
]
OBS_WINDOW = 10
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TRANSACTION_COST = 0.001  # 0.1%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QLOG_NAME = "qvals_log.csv"
TRADES_CSV = "inference_trades.csv"
PLOT_PNG = "inference_trades_plot_improved.png"

# ---------------- small model and environment classes ----------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

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

class TradingEnv:
    """
    Trading environment.
    - Uses Close_orig for pricing.
    - FEATURES columns in df should be the scaled versions for state inputs.
    - If fractional==True: buys fractional shares = cash / price.
    - If fractional==False (single-share mode): buys exactly 1 share if cash allows.
    """
    def __init__(self, df: pd.DataFrame, obs_window: int = OBS_WINDOW, transaction_cost: float = TRANSACTION_COST, fractional: bool = False):
        self.df = df.reset_index(drop=True)
        self.obs_window = obs_window
        self.transaction_cost = transaction_cost
        self.fractional = fractional
        self._max_index = len(df) - 1
        self.reset()

    def reset(self, start_index: int = None, starting_cash: float = 10000.0):
        if start_index is None:
            start_index = self.obs_window
        self.idx = start_index
        self.position = 0  # 0 flat, 1 long
        self.cash = float(starting_cash)
        self.shares = 0.0
        self.portfolio_value = float(self.cash)
        self.done = False
        # optional: track hold duration
        self.hold_steps = 0
        return self._get_state()

    def _get_price(self, index: int) -> float:
        return float(self.df.loc[index, "Close_orig"])

    def _get_state(self) -> np.ndarray:
        start = self.idx - self.obs_window
        window = self.df.loc[start:self.idx - 1, FEATURES].values
        if window.shape[0] < self.obs_window:
            pad = np.zeros((self.obs_window - window.shape[0], window.shape[1]))
            window = np.vstack([pad, window])
        flat = window.flatten().astype(np.float32)
        extras = np.array([self.position, self.cash, self.portfolio_value], dtype=np.float32)
        return np.concatenate([flat, extras])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: 0 hold, 1 buy, 2 sell
        Returns: next_state, reward, done, info
        """
        if self.done:
            raise RuntimeError("step called on terminated env")

        prev_price = self._get_price(self.idx - 1)
        cur_price = self._get_price(self.idx)

        executed = False
        trade_cost = 0.0

        if action == 1:  # Buy attempt
            if self.position == 0:
                if self.fractional:
                    # buy fractional shares equal to cash / price
                    self.shares = (self.cash / cur_price)
                    trade_cost = cur_price * self.shares * self.transaction_cost
                    # subtract price*shares + fee
                    self.cash -= (cur_price * self.shares + trade_cost)
                    self.position = 1
                    executed = True
                    self.hold_steps = 0
                else:
                    # single-share mode: buy 1 share if cash sufficient including fee
                    need = cur_price * 1.0 * (1.0 + self.transaction_cost)
                    if self.cash >= need:
                        self.shares = 1.0
                        trade_cost = cur_price * self.shares * self.transaction_cost
                        self.cash -= (cur_price * self.shares + trade_cost)
                        self.position = 1
                        executed = True
                        self.hold_steps = 0
                    else:
                        executed = False

        elif action == 2:  # Sell attempt
            if self.position == 1:
                proceeds = cur_price * self.shares
                trade_cost = proceeds * self.transaction_cost
                self.cash += (proceeds - trade_cost)
                self.shares = 0.0
                self.position = 0
                executed = True
                self.hold_steps = 0

        # Hold -> nothing (action == 0)
        # Update portfolio value
        holdings_value = cur_price * self.shares
        prev_portfolio = self.portfolio_value
        self.portfolio_value = self.cash + holdings_value
        reward = (self.portfolio_value - prev_portfolio)

        # advance index
        self.idx += 1
        if self.idx > self._max_index:
            self.done = True
        else:
            self.done = False
            if self.position == 1:
                self.hold_steps += 1

        next_state = self._get_state() if not self.done else np.zeros_like(self._get_state())

        info = {
            "executed": executed,
            "trade_cost": trade_cost,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "position": self.position,
            "price": cur_price,
            "hold_steps": getattr(self, "hold_steps", 0)
        }
        return next_state, float(reward), self.done, info

class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, lr=1e-4):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.policy_net = QNetwork(state_dim, n_actions).to(DEVICE)
        self.target_net = QNetwork(state_dim, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, explore=False) -> int:
        self.policy_net.eval()
        state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            qvals = self.policy_net(state_v)
        return int(torch.argmax(qvals).item())

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        if isinstance(ckpt, dict):
            if 'policy_state_dict' in ckpt:
                self.policy_net.load_state_dict(ckpt['policy_state_dict'])
            elif 'state_dict' in ckpt:
                self.policy_net.load_state_dict(ckpt['state_dict'])
            else:
                # fallback assume it's a state_dict
                self.policy_net.load_state_dict(ckpt)
            if 'target_state_dict' in ckpt:
                try:
                    self.target_net.load_state_dict(ckpt['target_state_dict'])
                except Exception:
                    pass
        else:
            self.policy_net.load_state_dict(ckpt)
        self.policy_net.eval()

# ---------------- preprocessing and helpers ----------------
def load_and_preprocess(csv_path: str, features=FEATURES, obs_window: int = OBS_WINDOW) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    df = df[features].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled, columns=features)
    df_scaled = df_scaled.rename(columns={c: f"{c}_s" for c in features})

    df_out = pd.concat([df.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)
    return df_out

def find_checkpoint(candidates: List[str], provided: str = None) -> str:
    if provided:
        if os.path.exists(provided):
            return provided
        else:
            raise FileNotFoundError(f"--checkpoint passed but file not found: {provided}")
    for fn in candidates:
        if os.path.exists(fn):
            return fn
    pths = [f for f in os.listdir('.') if f.endswith('.pth')]
    if pths:
        return pths[0]
    raise FileNotFoundError(f"No checkpoint found. Tried: {candidates} and found: {pths}")

def plot_trades(trades_df: pd.DataFrame, original_df: pd.DataFrame, out_png: str = PLOT_PNG):
    # x axis uses Date if present in original_df, otherwise simple index
    if 'Date' in original_df.columns:
        # align dates: original_df has rows for all timestamps; trades_df index corresponds to original_df rows from OBS_WINDOW onward
        date_index = original_df['Date'].iloc[OBS_WINDOW:OBS_WINDOW + len(trades_df)].reset_index(drop=True)
        x = pd.to_datetime(date_index)
        use_dates = True
    else:
        x = np.arange(len(trades_df))
        use_dates = False

    price = trades_df['price'].values
    pv = trades_df['portfolio_value'].values

    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax1.plot(x, price, label='Close price', linewidth=1)
    ax1.set_ylabel("Price")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(x, pv, label='Portfolio value', linewidth=1, linestyle='--')
    ax2.set_ylabel("Portfolio value")

    attempted_buys = trades_df[trades_df['action'] == 1]
    attempted_sells = trades_df[trades_df['action'] == 2]
    executed = trades_df[trades_df['executed'] == True]

    # Attempted markers low alpha
    if not attempted_buys.empty:
        ax1.scatter(x[attempted_buys.index], attempted_buys['price'], marker='^', s=30, alpha=0.25, label='Attempted Buys')
    if not attempted_sells.empty:
        ax1.scatter(x[attempted_sells.index], attempted_sells['price'], marker='v', s=30, alpha=0.25, label='Attempted Sells')

    # Executed trades bold
    if not executed.empty:
        ax1.scatter(x[executed.index], executed['price'], marker='o', s=90, c='red', label='Executed trades')

    ax1.set_title("Price and Portfolio Value with Agent Actions (attempted vs executed)")
    if use_dates:
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=30)
    else:
        ax1.xaxis.set_major_locator(plt.MaxNLocator(10))

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved improved plot to:", out_png)

# ---------------- main inference flow ----------------
def run_inference(csv_path: str, checkpoint: str, fractional: bool, starting_cash: float, debug_steps: int = 10):
    cwd = os.getcwd()
    print("Working folder:", cwd)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    ckpt_path = find_checkpoint(CHECKPOINT_CANDIDATES, provided=checkpoint)
    print("Using checkpoint:", ckpt_path)

    df = load_and_preprocess(csv_path)
    # save original close for pricing
    df_for_env = df.copy()
    df_for_env["Close_orig"] = df_for_env["Close"]

    # copy scaled columns into FEATURES (these are used by env._get_state)
    for c in FEATURES:
        df_for_env[c] = df_for_env[f"{c}_s"]

    env = TradingEnv(df_for_env, obs_window=OBS_WINDOW, transaction_cost=TRANSACTION_COST, fractional=fractional)
    state_dim = OBS_WINDOW * len(FEATURES) + 3
    n_actions = 3

    agent = DQNAgent(state_dim=state_dim, n_actions=n_actions)
    agent.load(ckpt_path)
    print("Model loaded.")

    # prepare qlog file
    if os.path.exists(QLOG_NAME):
        os.remove(QLOG_NAME)

    # reset environment with realistic starting cash
    state = env.reset(start_index=OBS_WINDOW, starting_cash=starting_cash)
    print(f"Starting cash: {env.cash:.2f} | fractional mode: {env.fractional}")

    # inference loop with qval logging
    records = []
    step = 0
    attempted_buys = attempted_sells = 0
    executed_total = executed_buys = executed_sells = 0

    with open(QLOG_NAME, "w") as qf:
        qf.write("step,q0,q1,q2,chosen\n")

    while True:
        # compute q-values for logging
        state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            qvals = agent.policy_net(state_v).cpu().numpy().flatten()
        chosen = int(np.argmax(qvals))

        # log qvals
        with open(QLOG_NAME, "a") as qf:
            qf.write(f"{step},{qvals[0]},{qvals[1]},{qvals[2]},{chosen}\n")

        # count attempted actions
        if chosen == 1:
            attempted_buys += 1
        elif chosen == 2:
            attempted_sells += 1

        # debug print for first N steps to inspect values
        if step < debug_steps:
            cur_price = env._get_price(env.idx)
            need = cur_price * (1.0 + env.transaction_cost)
            print(f"[DBG] step={step} idx={env.idx} price={cur_price:.2f} need={need:.2f} cash={env.cash:.2f} pos={env.position} qvals={qvals} action={chosen}")

        next_state, reward, done, info = env.step(chosen)

        if info.get('executed', False):
            executed_total += 1
            if chosen == 1:
                executed_buys += 1
            elif chosen == 2:
                executed_sells += 1

        records.append({
            "step": step,
            "index": env.idx - 1,
            "price": info.get('price'),
            "action": chosen,
            "action_str": ("Hold", "Buy", "Sell")[chosen],
            "executed": info.get('executed', False),
            "trade_cost": info.get('trade_cost', 0.0),
            "portfolio_value": info.get('portfolio_value'),
            "cash": info.get('cash'),
            "shares": info.get('shares'),
            "position": info.get('position'),
            "hold_steps": info.get('hold_steps', 0)
        })

        state = next_state
        step += 1

        if done:
            break

    trades_df = pd.DataFrame(records)
    trades_df.to_csv(TRADES_CSV, index=False)
    print(f"Saved trades to {os.path.join(cwd, TRADES_CSV)}")
    print(f"Saved Q-values log to {os.path.join(cwd, QLOG_NAME)}")

    # summary
    total_steps = len(trades_df)
    holds = trades_df[trades_df['action'] == 0].shape[0]
    final_pv = env.portfolio_value
    print("\n===== SUMMARY =====")
    print(f"Steps (timesteps processed): {total_steps}")
    print(f"Attempted actions -> Buys: {attempted_buys}, Sells: {attempted_sells}, Holds: {holds}")
    print(f"Executed trades -> Total: {executed_total}, Buys executed: {executed_buys}, Sells executed: {executed_sells}")
    print(f"Final portfolio value (starting cash {starting_cash:.2f}): {final_pv:.6f}")

    # improved plotting
    try:
        plot_trades(trades_df, df, out_png=PLOT_PNG)
    except Exception as e:
        print("Plotting failed:", e)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="DQN inference runner with refined options.")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="CSV file path (default: AMZN.csv)")
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (.pth). If not set, script tries defaults.")
    p.add_argument("--fractional", action="store_true", help="Use fractional shares (default False -> single-share).")
    p.add_argument("--starting-cash", type=float, default=10000.0, help="Starting capital for inference (default 10000.0).")
    p.add_argument("--debug-steps", type=int, default=10, help="Number of initial debug prints for inspection.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(csv_path=args.csv, checkpoint=args.checkpoint, fractional=args.fractional, starting_cash=args.starting_cash, debug_steps=args.debug_steps)
