# DQN Trading Agent for Single-Stock Trading

> A Deep Q-Learning implementation that trains a discrete action agent (Hold / Buy / Sell) on historical OHLCV data. The agent uses a Double-DQN architecture with experience replay and a custom trading environment that simulates fractional-share long-only trading with transaction costs.

## Project Overview

This project implements a reinforcement learning pipeline to learn a trading policy from historical stock data. The agent observes a sliding window of scaled OHLCV data plus simple portfolio state (position, cash, portfolio value) and chooses between three discrete actions:

* `0` = Hold
* `1` = Buy (use available cash to buy fractional shares)
* `2` = Sell (sell all holdings)

Training uses Double-DQN with a separate target network, experience replay, and epsilon-greedy exploration.


## Key Concepts & Components

* **ReplayBuffer**: stores transitions `(state, action, reward, next_state, done)` and samples random minibatches for training.
* **QNetwork**: feed-forward MLP that maps state vectors to Q-values for each action.
* **TradingEnv**: custom environment stepping through the OHLCV CSV, simulating portfolio updates and computing step rewards as change in portfolio value.
* **DQNAgent**: holds policy and target networks, optimizer, training logic (Double-DQN), epsilon schedule, and model saving/loading.

## Data and Preprocessing

* The code expects a CSV file containing the columns: `Open, High, Low, Close, Volume`. If a `Date` column exists, it is sorted by date.
* A `StandardScaler` is fit to the full dataset and used to transform OHLCV into scaled features (appended as columns ended with `_s`). The scaled columns are copied back into the expected feature names before creating environment data.

**Note about leakage:** fitting a scaler on the entire dataset introduces look-ahead/data leakage. See "Known Issues" for a fix.

## Trading Environment

* **State**: Flattened vector of `OBS_WINDOW * n_features` scaled values + 3 extras: `position` (0/1), `cash`, `portfolio_value`.
* **Actions**: Hold / Buy / Sell.
* **Positioning**: long-only with fractional shares allowed. Buying uses current cash to purchase fractional shares; selling liquidates all shares.
* **Transaction cost**: a proportional fee per trade (default `0.001` = 0.1%).
* **Reward**: `reward = new_portfolio_value - old_portfolio_value` (stepwise change in portfolio value).

## DQN Agent & Training

* **Network**: two hidden layers (128 units each) with ReLU activations.
* **Double DQN**: the policy network selects the next action while the target network evaluates the next state's Q-value.
* **Replay**: a large replay buffer (default 100k) with a warmup of `MIN_REPLAY_SIZE` before training begins.
* **Epsilon decay**: linear decay from `EPS_START` to `EPS_END` over `EPS_DECAY_STEPS` steps.
* **Target updates**: policy weights are copied to target network every `TARGET_UPDATE_FREQ` steps.
* **Saving**: the best model (by final portfolio value) is saved and periodic checkpoints are created.

## Hyperparameters (defaults)

These are defined at the top of the script; common ones:

* `OBS_WINDOW = 10`
* `BATCH_SIZE = 64`
* `GAMMA = 0.99`
* `LR = 1e-4`
* `REPLAY_BUFFER_SIZE = 100000`
* `MIN_REPLAY_SIZE = 2000`
* `TARGET_UPDATE_FREQ = 1000`
* `MAX_EPISODES = 200`
* `EPS_START = 1.0`, `EPS_END = 0.02`, `EPS_DECAY_STEPS = 20000`
* `TRANSACTION_COST = 0.001`

Tune these to experiment with learning stability and speed.

## Known Issues & Suggested Fixes

1. **Calling `_get_state()` after end-of-data when creating a zero-shaped `next_state`**

   * The current code uses `np.zeros_like(self._get_state())` after stepping past the last index. Since `_get_state()` assumes `self.idx` is within range, this can raise an out-of-bounds error. Fix by precomputing the expected state shape or returning a zeros array of the known dimension.

   ```py
   # Fix: compute shape without indexing into df
   state_dim = self.obs_window * len(FEATURES) + 3
   next_state = np.zeros(state_dim, dtype=np.float32)
   ```

2. **Data leakage from scaling**

   * `StandardScaler` is fit on the entire dataset. Instead, split into train/test first and fit the scaler on the training split only, or use a rolling/online scaler. This avoids future-data information leaking into training.

3. **No explicit CLI args for CSV path**

   * The code uses a hard-coded `CSV_PATH` constant. You can either edit this constant or add a small `argparse` block to pass the CSV path and hyperparameters at runtime.

   Example snippet to add to the bottom of `train.py`:

   ```py
   if __name__ == "__main__":
       import argparse
       parser = argparse.ArgumentParser()
       parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to OHLCV CSV file")
       parser.add_argument("--episodes", type=int, default=MAX_EPISODES)
       parser.add_argument("--save", type=str, default=MODEL_SAVE_PATH)
       args = parser.parse_args()

       CSV_PATH = args.csv
       MAX_EPISODES = args.episodes
       MODEL_SAVE_PATH = args.save

       train()
   ```

## How to Run (Step-by-step)

Follow these steps to run the training script locally.

### 1) Clone the repo

```bash
git clone <your-repo-url>
cd DQN-Trading-Agent
```

### 2) Create a Python virtual environment

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# or Windows (cmd)
venv\Scripts\activate.bat
```

### 3) Install dependencies

Create `requirements.txt` with at least the following (example):

```
pandas
numpy
scikit-learn
torch
matplotlib
```

Then install:

```bash
pip install -r requirements.txt
```

> If you plan to use GPU acceleration, install a PyTorch build that matches your CUDA version. See the official PyTorch site for the right `pip`/`conda` command.

### 4) Prepare your data

* Place your OHLCV CSV (with `Open,High,Low,Close,Volume`) in `data/AMZN.csv` or anywhere and update `CSV_PATH` at the top of `train.py`.
* Alternatively, use the CLI snippet above to pass `--csv path/to/your.csv` (recommended).

### 5) Run training

If you did not add CLI arg support, edit `CSV_PATH` at the top of `train.py` to point to your CSV and then run:

```bash
python train.py
```

If you added the `argparse` example above, run:

```bash
python train.py --csv data/AMZN.csv --episodes 200 --save models/dqn_amzn.pth
```

### 6) Outputs

* Trained model(s) will be saved to `MODEL_SAVE_PATH` (default `dqn_amzn.pth`) and `best_dqn_amzn.pth` (if you keep the `best_` naming in the code).
* You should see epoch/episode printouts showing `Episode`, `Steps`, `EpReward`, `Final PV`, `Eps`, and `Replay` size.

## Reproducibility & Tips

* The script seeds `numpy`, `random`, and `torch` at the top. For full determinism, you may need to configure PyTorch flags (e.g., `torch.use_deterministic_algorithms(True)`) and control multi-threading behavior.
* If GPU memory is small, reduce `BATCH_SIZE` or use CPU-only training.
* Monitor learning with simple logs: add `matplotlib` plots for loss and portfolio value or save JSON logs for analysis.

## Further Improvements

* Use a train/test split and perform out-of-sample backtesting.
* Add feature engineering: returns, moving averages, RSI, Bollinger Bands.
* Add risk-aware reward terms (penalize drawdown, add Sharpe ratio regularizer).
* Try prioritized replay, dueling DQN, or actor-critic methods (SAC, PPO) for continuous sizing.
* Add transaction slippage, discrete lot sizes, or position sizing constraints for realism.

## License & Contact

This project is provided for educational purposes. 

