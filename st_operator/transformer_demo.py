"""
transformer_prob_return.py

功能：
- 用 Transformer Encoder 从过去 L 天的特征预测“明天上涨概率”和“明天收益率期望”；
- 初始用前 initial_train_days 训练一次，随后按天滚动：预测 -> 获取真实 -> 将样本放入 replay buffer -> 用小批次（新样本 + 回放）微调模型（incremental style）。
- 支持时间衰减样本权重（指数衰减）。
- 数据要求（CSV列）： date, open, close, high, low, volume, cost_p5, cost_p15, cost_p50, cost_p85, cost_p95, weighted_cost, win_rate
"""

import os
import math
import random
from collections import deque
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DATA_FILE = "data.csv"   # if not present, script will synth data
SEQ_LEN = 30             # 用前 30 天数据预测第 31 天
INITIAL_TRAIN_DAYS = 180
BATCH_SIZE = 32
EPOCHS_INIT = 30
FINE_TUNE_STEPS = 20    # 每日微调步数
FINE_TUNE_BATCH = 32
LEARNING_RATE_INIT = 1e-3
LR_FINE_TUNE = 1e-5
MEMORY_MAXLEN = 1000
TIME_DECAY_LAMBDA = 0.005  # 指数衰减系数 (可调)
CLASS_WEIGHT = 1.0         # 分类 loss 权重
REG_WEIGHT = 1.0           # 回归 loss 权重

# -----------------------
# Data loading / synthetic
# -----------------------
expected_cols = [
    "date", "open", "close", "high", "low", "volume",
    "cost_p5", "cost_p15", "cost_p50", "cost_p85", "cost_p95",
    "weighted_cost", "win_rate"
]

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"data.csv missing columns: {missing}")
    df = df.sort_values("date").reset_index(drop=True)
else:
    # 生成合成数据方便 demo
    n = 400
    start = pd.Timestamp("2024-01-01")
    dates = [start + timedelta(days=i) for i in range(n)]
    price = 100 + np.cumsum(np.random.normal(0, 1.2, size=n))
    high = price + np.abs(np.random.normal(0, 0.8, size=n))
    low = price - np.abs(np.random.normal(0, 0.8, size=n))
    openp = price + np.random.normal(0, 0.5, size=n)
    close = price + np.random.normal(0, 0.6, size=n)
    volume = np.random.randint(500, 5000, size=n)
    cost_p5 = price - np.random.uniform(1, 3, size=n)
    cost_p15 = price - np.random.uniform(0.5, 2, size=n)
    cost_p50 = price + np.random.uniform(-0.5, 0.5, size=n)
    cost_p85 = price + np.random.uniform(0.5, 2, size=n)
    cost_p95 = price + np.random.uniform(1, 3, size=n)
    weighted_cost = (cost_p5 + cost_p15 + cost_p50 + cost_p85 + cost_p95) / 5
    win_rate = np.clip(0.4 + np.random.normal(0, 0.05, size=n), 0, 1)
    df = pd.DataFrame({
        "date": dates, "open": openp, "close": close, "high": high, "low": low, "volume": volume,
        "cost_p5": cost_p5, "cost_p15": cost_p15, "cost_p50": cost_p50, "cost_p85": cost_p85, "cost_p95": cost_p95,
        "weighted_cost": weighted_cost, "win_rate": win_rate
    })

df = df[expected_cols].sort_values("date").reset_index(drop=True)

# -----------------------
# Feature engineering helper
# -----------------------
def build_features(df):
    X = df.copy()
    # simple returns and pct changes, lags
    X["return"] = X["close"].pct_change().fillna(0)
    X["hl_range"] = (X["high"] - X["low"]) / X["open"]
    for lag in [1,2,3,5]:
        X[f"ret_lag_{lag}"] = X["return"].shift(lag).fillna(0)
        X[f"vol_lag_{lag}"] = X["volume"].shift(lag).fillna(method="bfill")
    # percent distance to cost percentiles
    for c in ["cost_p5","cost_p15","cost_p50","cost_p85","cost_p95","weighted_cost"]:
        X[f"pd_{c}"] = (X["close"] - X[c]) / X[c]
    X["weekday"] = X["date"].dt.weekday
    # drop NA (early rows)
    X = X.fillna(method="bfill").reset_index(drop=True)
    return X

df_feat = build_features(df)

# -----------------------
# Dataset: sliding windows
# -----------------------
class SlidingWindowDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.feature_cols = [c for c in df.columns if c not in ("date","close")]
        # target: tomorrow's return and up label
        self.targets = self._build_targets()

    def _build_targets(self):
        # next-day return
        ret = self.df["close"].pct_change().shift(-1).fillna(0)
        # binary up prob: 1 if next-day return > 0 else 0
        up = (ret > 0).astype(float)
        return pd.DataFrame({"ret_next": ret, "up_next": up})

    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        # window [idx, idx+seq_len-1] -> predict idx+seq_len (next day)
        start = idx
        end = idx + self.seq_len
        Xwin = self.df.loc[start:end-1, self.feature_cols].values.astype(np.float32)  # seq_len x f
        target_row = self.targets.iloc[end]  # next day
        sample = {
            "X": torch.tensor(Xwin, dtype=torch.float32),
            "ret": torch.tensor(float(target_row["ret_next"]), dtype=torch.float32),
            "up": torch.tensor(float(target_row["up_next"]), dtype=torch.float32),
            "date": self.df.loc[end, "date"]
        }
        return sample

# -----------------------
# Model: Transformer Encoder with dual heads
# -----------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        # use learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # optional adapter pattern could be inserted here (omitted for brevity)
        # pooling + heads
        self.pool = nn.AdaptiveAvgPool1d(1)  # applied on (batch, seq, d) via permute
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        h = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
        h_enc = self.encoder(h)  # (batch, seq, d)
        # global pooling over seq
        h_pooled = h_enc.permute(0,2,1)  # (batch, d, seq)
        h_pooled = self.pool(h_pooled).squeeze(-1)  # (batch, d)
        ret_pred = self.reg_head(h_pooled).squeeze(-1)   # continuous
        up_prob = self.cls_head(h_pooled).squeeze(-1)    # [0,1]
        return up_prob, ret_pred

# -----------------------
# Helpers: sample weights (time decay)
# -----------------------
def compute_time_weights(dates, current_date, lam=TIME_DECAY_LAMBDA):
    # dates: list/array of pd.Timestamp for each sample
    # returns numpy array of weights same length
    ages = np.array([(current_date - d).days for d in dates], dtype=np.float32)
    weights = np.exp(-lam * ages)
    return weights

# -----------------------
# Training / Online loop
# -----------------------
def collate_fn(batch):
    X = torch.stack([b["X"] for b in batch], dim=0)
    ret = torch.stack([b["ret"] for b in batch], dim=0)
    up = torch.stack([b["up"] for b in batch], dim=0)
    dates = [b["date"] for b in batch]
    return {"X": X, "ret": ret, "up": up, "dates": dates}

# prepare dataset
dataset = SlidingWindowDataset(df_feat, seq_len=SEQ_LEN)
total_len = len(dataset)
print(f"Total sliding samples: {total_len}")

# initial train indices: 0 .. initial_train_days - seq_len - 1 maybe; we compute split by date
# We'll pick initial training samples whose target date index < INITIAL_TRAIN_DAYS
# find sample index i where target date index (i+seq_len) < INITIAL_TRAIN_DAYS_index
initial_end_index = max(INITIAL_TRAIN_DAYS - SEQ_LEN - 1, 0)
initial_indices = list(range(0, initial_end_index+1))
stream_indices = list(range(initial_end_index+1, total_len))

print(f"Initial train samples: {len(initial_indices)}, streaming samples: {len(stream_indices)}")

# DataLoader for initial train
def make_loader_from_indices(indices, batch_size=BATCH_SIZE, shuffle=True):
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# create model
num_features = len(dataset.feature_cols)
model = TimeSeriesTransformer(num_features=num_features, d_model=64, nhead=4, num_layers=2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_INIT)
bce_loss = nn.BCELoss(reduction="none")   # we'll apply sample weights externally
mse_loss = nn.MSELoss(reduction="none")

# initial training
if len(initial_indices) > 0:
    loader_init = make_loader_from_indices(initial_indices, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS_INIT):
        model.train()
        epoch_losses = []
        for batch in loader_init:
            Xb = batch["X"].to(DEVICE)
            ret_true = batch["ret"].to(DEVICE)
            up_true = batch["up"].to(DEVICE)
            dates = batch["dates"]
            # compute current_date as last date in batch's window target (max of dates)
            current_date = max(dates)
            weights = compute_time_weights(dates=dates, current_date=current_date, lam=TIME_DECAY_LAMBDA)
            weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

            up_pred, ret_pred = model(Xb)
            loss_cls = bce_loss(up_pred, up_true)
            loss_reg = mse_loss(ret_pred, ret_true)
            # combine with per-sample weights
            loss = (CLASS_WEIGHT * (loss_cls * weights).mean()) + (REG_WEIGHT * (loss_reg * weights).mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"[Init train] Epoch {epoch+1}/{EPOCHS_INIT}, loss={np.mean(epoch_losses):.6f}")

# prepare replay buffer (store dicts: {"X": np.array, "ret":float, "up":float, "date":pd.Timestamp})
memory = deque(maxlen=MEMORY_MAXLEN)

# fill memory initially with a subset of training data
def push_initial_memory(indices, max_items=MEMORY_MAXLEN//2):
    chosen = random.sample(indices, min(len(indices), max_items))
    for idx in chosen:
        sample = dataset[idx]
        memory.append({
            "X": sample["X"].numpy(),
            "ret": float(sample["ret"].item()),
            "up": float(sample["up"].item()),
            "date": sample["date"]
        })
push_initial_memory(initial_indices)

# streaming prediction + incremental fine-tune
results = []
model.eval()
for idx in stream_indices:
    # 1) predict for sample idx
    sample = dataset[idx]  # corresponds to target at date = sample["date"]
    X_in = sample["X"].unsqueeze(0).to(DEVICE)  # (1, seq, f)
    with torch.no_grad():
        up_prob, ret_pred = model(X_in)
        up_p = float(up_prob.item())
        ret_p = float(ret_pred.item())
    predict_date = sample["date"]  # predicted date (target date)
    # the true labels are available in dataset.targets at idx+seq_len
    # but in our dataset sample["ret"], sample["up"] are already the true next-day values (for simulation)
    y_ret = float(sample["ret"].item())
    y_up = float(sample["up"].item())

    results.append({
        "date": predict_date,
        "pred_up_prob": up_p,
        "pred_ret": ret_p,
        "true_up": y_up,
        "true_ret": y_ret
    })

    # 2) online update: add this sample to memory, then fine-tune with small batches (new + replay)
    memory.append({
        "X": sample["X"].numpy(),
        "ret": y_ret,
        "up": y_up,
        "date": sample["date"]
    })

    # Fine-tune small steps
    if len(memory) > 0:
        model.train()
        opt_finetune = optim.Adam(model.parameters(), lr=LR_FINE_TUNE)
        for step in range(FINE_TUNE_STEPS):
            # sample a batch from memory (mix new and old)
            batch_samples = random.sample(list(memory), min(len(memory), FINE_TUNE_BATCH))
            Xb = torch.tensor(np.stack([s["X"] for s in batch_samples]), dtype=torch.float32).to(DEVICE)
            rets = torch.tensor([s["ret"] for s in batch_samples], dtype=torch.float32).to(DEVICE)
            ups = torch.tensor([s["up"] for s in batch_samples], dtype=torch.float32).to(DEVICE)
            dates_batch = [s["date"] for s in batch_samples]
            current_date = max(dates_batch)
            weights = compute_time_weights(dates=dates_batch, current_date=current_date, lam=TIME_DECAY_LAMBDA)
            weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

            up_pred_b, ret_pred_b = model(Xb)
            loss_cls = bce_loss(up_pred_b, ups)
            loss_reg = mse_loss(ret_pred_b, rets)
            loss = (CLASS_WEIGHT * (loss_cls * weights).mean()) + (REG_WEIGHT * (loss_reg * weights).mean())

            opt_finetune.zero_grad()
            loss.backward()
            opt_finetune.step()
        model.eval()

# -----------------------
# Evaluation & Save
# -----------------------
res_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
# metrics
auc = roc_auc_score(res_df["true_up"], res_df["pred_up_prob"]) if len(res_df["true_up"].unique()) > 1 else float("nan")
ll = log_loss(res_df["true_up"], np.clip(res_df["pred_up_prob"], 1e-6, 1-1e-6)) if len(res_df["true_up"].unique()) > 1 else float("nan")
rmse = math.sqrt(mean_squared_error(res_df["true_ret"], res_df["pred_ret"]))
mae = mean_absolute_error(res_df["true_ret"], res_df["pred_ret"])

print("Streaming Eval:")
print(f"Samples: {len(res_df)}  AUC: {auc:.4f}  LogLoss: {ll:.6f}  RMSE(ret): {rmse:.6f}  MAE(ret): {mae:.6f}")

res_df.to_csv("transformer_prob_return_results.csv", index=False)
print("Saved results to transformer_prob_return_results.csv")
