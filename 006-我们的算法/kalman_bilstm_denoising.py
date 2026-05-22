# 时间序列传感器去噪 —— Kalman 滤波 + BiLSTM 残差修正联合框架
#
# 算法原理（两阶段）：
#   阶段一 · Kalman 滤波（线性最优估计）
#     状态方程：x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
#     观测方程：z_t = x_t  + v_t,      v_t ~ N(0, R)
#     对线性高斯噪声给出理论最优解，参数 Q/R 从训练集自动估计
#
#   阶段二 · BiLSTM 残差修正（非线性补偿）
#     输入：[噪声列(6) + Kalman预测(6)] = 12 维时序特征
#     输出：residual = true_value - kalman_pred（每特征独立模型）
#     最终预测 = Kalman预测 + BiLSTM残差修正
#
#   理论依据：Deep Kalman Filters (Krishnan et al., 2015)
#             Recurrent Kalman Networks (Becker et al., ICML 2019)

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

start_time = time.time()

# ─── LSTM 超参数 ───────────────────────────────────────────────────────────────
WINDOW      = 30
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
BATCH_SIZE  = 1024
EPOCHS      = 100
LR          = 5e-4
PATIENCE    = 15
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = torch.cuda.is_available()

# ─── 数据路径 ──────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
TRAIN_PATH  = str(_ROOT / "001-数据集/ML期末数据集（含真实值）/modified_数据集Time_Series661.dat")
TEST_PATH   = str(_ROOT / "001-数据集/ML期末数据集（不含真实值）/modified_数据集Time_Series662.dat")
OUTPUT_PATH = str(Path(__file__).resolve().parent / "result_Kalman_BiLSTM.csv")

# ─── 特征列 ────────────────────────────────────────────────────────────────────
TARGET_COLS = [
    'T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
    'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth'
]
NOISE_COLS = [
    'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
    'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth'
]


# ══════════════════════════════════════════════════════════════════════════════
# 阶段一：Kalman 滤波
# ══════════════════════════════════════════════════════════════════════════════

def kalman_filter_1d(observations: np.ndarray, Q: float, R: float) -> np.ndarray:
    n = len(observations)
    x = observations[0]
    P = R
    estimates = np.empty(n)
    for i, z in enumerate(observations):
        P = P + Q
        K = P / (P + R)
        x = x + K * (z - x)
        P = (1.0 - K) * P
        estimates[i] = x
    return estimates


def estimate_params(train_df: pd.DataFrame):
    params = {}
    for target, noise in zip(TARGET_COLS, NOISE_COLS):
        true_vals  = train_df[target].values
        noisy_vals = train_df[noise].values
        R = float(np.var(noisy_vals - true_vals))
        Q = float(np.var(np.diff(true_vals)))
        params[target] = (Q, R)
    return params


def run_kalman(df: pd.DataFrame, params: dict) -> np.ndarray:
    preds = []
    for target, noise in zip(TARGET_COLS, NOISE_COLS):
        Q, R = params[target]
        preds.append(kalman_filter_1d(df[noise].values, Q, R))
    return np.stack(preds, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 阶段二：BiLSTM 残差修正
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        return self.X[idx: idx + self.window], self.y[idx + self.window]


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.score(lstm_out), dim=1)
        return (weights * lstm_out).sum(dim=1)


class BiLSTMResidual(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        context = self.attention(out)
        return self.head(context).squeeze(-1)


def train_residual_model(model, train_loader, val_loader):
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5)
    criterion  = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val, best_state, no_improve = float('inf'), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                loss = criterion(model(xb), yb)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>4}  Train={train_loss:.6f}  Val={val_loss:.6f}  "
                  f"LR={optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    早停 (epoch={epoch}, best_val={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Kalman 滤波 + BiLSTM 残差修正  联合去噪框架")
    print(f"  设备: {DEVICE}   窗口: {WINDOW}   隐层: {HIDDEN_SIZE}x2(双向)")
    print("=" * 65)

    # 1. 加载数据
    print("\n[1/5] 加载数据集...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  训练集: {len(train_df):,} 行   测试集: {len(test_df):,} 行")

    # 2. Kalman 滤波
    print("\n[2/5] 阶段一：Kalman 滤波...")
    params = estimate_params(train_df)
    print(f"  {'通道':<30} {'Q':<14} {'R':<14} {'稳态增益K'}")
    for target, (Q, R) in params.items():
        K = Q / (Q + R) if (Q + R) > 0 else 0
        print(f"  {target:<30} {Q:<14.6f} {R:<14.6f} {K:.4f}")

    kalman_train = run_kalman(train_df, params)
    kalman_test  = run_kalman(test_df,  params)
    true_train   = train_df[TARGET_COLS].values
    noise_train  = train_df[NOISE_COLS].values
    noise_test   = test_df[NOISE_COLS].values

    kalman_mae = np.mean(np.abs(kalman_train - true_train))
    print(f"\n  Kalman 训练集总体 MAE: {kalman_mae:.4f}")

    # 3. 构造 LSTM 输入
    print("\n[3/5] 阶段二：构造残差学习输入（12维 = 噪声6 + Kalman预测6）...")
    X_train_raw    = np.hstack([noise_train, kalman_train])
    X_test_raw     = np.hstack([noise_test,  kalman_test ])
    residual_train = true_train - kalman_train

    scaler_X = StandardScaler().fit(X_train_raw)
    X_train  = scaler_X.transform(X_train_raw)
    X_test   = scaler_X.transform(X_test_raw)

    split   = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:split], X_train[split:]

    # 4. 每特征独立训练残差 BiLSTM
    print("\n[4/5] 逐特征训练 BiLSTM 残差修正模型...")
    residual_test_preds = []

    for i, target in enumerate(TARGET_COLS):
        print(f"\n  -- [{i+1}/6] {target} --")
        res_col    = residual_train[:, i:i+1]
        scaler_res = StandardScaler().fit(res_col[:split])
        res_tr     = scaler_res.transform(res_col[:split]).ravel()
        res_va     = scaler_res.transform(res_col[split:]).ravel()

        train_ds = TimeSeriesDataset(X_tr, res_tr, WINDOW)
        val_ds   = TimeSeriesDataset(X_va, res_va, WINDOW)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=USE_AMP)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=USE_AMP)
        print(f"    训练样本: {len(train_ds):,}   验证样本: {len(val_ds):,}")
        print(f"    残差范围: [{res_col.min():.4f}, {res_col.max():.4f}]  均值={res_col.mean():.6f}")

        model = BiLSTMResidual(
            input_size=12,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)
        model = train_residual_model(model, train_loader, val_loader)

        X_ctx   = np.vstack([X_train[-WINDOW:], X_test])
        y_dummy = np.zeros(len(X_ctx))
        test_ds     = TimeSeriesDataset(X_ctx, y_dummy, WINDOW)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=USE_AMP)
        test_res_norm = []
        model.eval()
        with torch.no_grad():
            for xb, _ in test_loader:
                test_res_norm.append(model(xb.to(DEVICE)).cpu().numpy())
        test_res = scaler_res.inverse_transform(
            np.concatenate(test_res_norm).reshape(-1, 1)).ravel()
        residual_test_preds.append(test_res)

    # 5. 最终预测 = Kalman + 残差修正
    print("\n[5/5] 合并输出：Kalman预测 + BiLSTM残差修正...")
    residual_matrix = np.stack(residual_test_preds, axis=1)
    final_preds     = kalman_test + residual_matrix

    pred_strings = [' '.join(map(str, row)) for row in final_preds]
    result_df    = pd.DataFrame({'Predicted_Value': pred_strings})
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"  预测结果已保存至 {OUTPUT_PATH}（共 {len(result_df):,} 行）")

    elapsed = time.time() - start_time
    print(f"\n总耗时：{elapsed:.2f} 秒")
    print("=" * 65)


if __name__ == '__main__':
    main()
