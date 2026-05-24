# 时间序列传感器去噪 —— 时序卷积网络（TCN）每特征独立训练
#
# 算法原理：
#   TCN 使用膨胀因果卷积（Dilated Causal Convolution）替代 LSTM 的循环结构：
#     - 因果性：t 时刻只能看到 t 及之前的数据（无未来信息泄露）
#     - 膨胀卷积：第 k 层膨胀率 d=2^k，感受野 = 2^(层数) × kernel_size
#       4层 × kernel=4 → 感受野 = 64 步
#     - 残差连接：每个 TCN Block 包含残差跳跃连接，训练更稳定
#     - 并行计算：无时序依赖，GPU 利用率远高于 LSTM
#
#   输入特征（17维）：
#     · 6维噪声列（Error_*）
#     · 气象变量因日间分布偏移大已移除
#     · 4维时间编码（hour_sin/cos 日变化、doy_sin/cos 季节变化）
#
#   参考论文：Bai et al., "An Empirical Evaluation of Generic Convolutional
#             and Recurrent Networks for Sequence Modeling", 2018 (arXiv:1803.01271)

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

start_time = time.time()

# ─── 全局超参数 ────────────────────────────────────────────────────────────────
WINDOW       = 64
NUM_CHANNELS = [64, 64, 64, 64]
KERNEL_SIZE  = 4
DROPOUT      = 0.2
BATCH_SIZE   = 1024
EPOCHS       = 200
LR           = 5e-4
PATIENCE     = 20
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP      = torch.cuda.is_available()

# ─── 每特征独立超参数 ──────────────────────────────────────────────────────────
PER_FEATURE_PARAMS = {
    'CO2_density':           {'PATIENCE': 25},
    'CO2_density_fast_tmpr': {'PATIENCE': 25},
}

# ─── 数据路径 ──────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
TRAIN_PATH  = str(_ROOT / "001-数据集/ML期末数据集（含真实值）/modified_数据集Time_Series661.dat")
TEST_PATH   = str(_ROOT / "001-数据集/ML期末数据集（不含真实值）/modified_数据集Time_Series662.dat")
OUTPUT_PATH = str(Path(__file__).resolve().parent / "result_TCN.csv")
MODEL_DIR   = Path(__file__).resolve().parent / "tcn_models"   # 模型保存目录

# ─── 特征列 ────────────────────────────────────────────────────────────────────
TARGET_COLS = [
    'T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
    'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth'
]
NOISE_COLS = [
    'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
    'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth'
]
# 额外物理特征（训练集和测试集均有）
EXTRA_COLS = []   # 气象变量每天分布不同，训练/测试集分布偏移大，去掉


# ══════════════════════════════════════════════════════════════════════════════
# 特征工程
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    构造完整输入特征矩阵（17维）：
      · 6维噪声列
      · 7维物理量
      · 4维时间周期编码（sin/cos）
    """
    # 噪声列 + 物理量
    base = df[NOISE_COLS + EXTRA_COLS].values.astype(np.float64)

    # 时间周期编码
    ts = pd.to_datetime(df['TIMESTAMP'], format='mixed')
    hour = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    doy  = ts.dt.day_of_year.astype(float)

    hour_sin = np.sin(2 * np.pi * hour / 24).values
    hour_cos = np.cos(2 * np.pi * hour / 24).values
    doy_sin  = np.sin(2 * np.pi * doy  / 365).values
    doy_cos  = np.cos(2 * np.pi * doy  / 365).values

    time_feats = np.stack([hour_sin, hour_cos, doy_sin, doy_cos], axis=1)

    return np.hstack([base, time_feats])   # (n, 17)


# ══════════════════════════════════════════════════════════════════════════════
# TCN 网络结构
# ══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """带因果填充的 1D 卷积：确保 t 时刻只能看到 t 及之前的数据"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x


class TCNBlock(nn.Module):
    """TCN 残差块：2层膨胀因果卷积 + 残差连接"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out).transpose(1, 2)).transpose(1, 2))
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNDenoiser(nn.Module):
    """
    TCN 去噪模型：
      输入: (batch, window, n_features)
      输出: (batch,) 单特征去噪值
    """
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 1),
        )

    def forward(self, x):
        out = self.network(x.transpose(1, 2))
        out = out[:, :, -1]
        return self.head(out).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 数据集
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


# ══════════════════════════════════════════════════════════════════════════════
# 训练函数
# ══════════════════════════════════════════════════════════════════════════════

def train_one(model, train_loader, val_loader, lr, patience):
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=7, factor=0.5)
    criterion  = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val, best_state, no_improve = float('inf'), None, 0

    print(f"    {'Epoch':<8} {'Train MSE':<16} {'Val MSE':<16} {'LR'}")
    print("    " + "-" * 54)

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
            print(f"    {epoch:<8} {train_loss:<16.6f} {val_loss:<16.6f} "
                  f"{optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    早停触发 (epoch={epoch}，最佳 Val MSE={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    n_layers  = len(NUM_CHANNELS)
    receptive = (KERNEL_SIZE - 1) * sum(2**i for i in range(n_layers)) + 1
    n_feats   = len(NOISE_COLS) + len(EXTRA_COLS) + 4   # 6+0+4=10
    print("=" * 65)
    print("  时序卷积网络（TCN）  【每特征独立训练 · 噪声+时间编码输入】")
    print(f"  设备: {DEVICE}   窗口: {WINDOW}   感受野: {receptive} 步")
    print(f"  输入维度: {n_feats}  (噪声6 + 时间编码4)")
    print(f"  层结构: {NUM_CHANNELS}   kernel={KERNEL_SIZE}")
    print("=" * 65)

    # 0. 创建模型保存目录
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. 加载数据并构造特征
    print("\n[1/4] 加载数据集并构造输入特征...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  训练集: {len(train_df):,} 行   测试集: {len(test_df):,} 行")

    X_train_raw = build_features(train_df)   # (n_train, 17)
    X_test_raw  = build_features(test_df)    # (n_test,  17)
    y_train_raw = train_df[TARGET_COLS].values.astype(np.float64)
    print(f"  特征矩阵: 训练 {X_train_raw.shape}   测试 {X_test_raw.shape}")

    # 2. 输入归一化
    print("\n[2/4] 输入特征归一化 (StandardScaler)...")
    scaler_X = StandardScaler().fit(X_train_raw)
    X_train  = scaler_X.transform(X_train_raw)
    X_test   = scaler_X.transform(X_test_raw)
    joblib.dump(scaler_X, MODEL_DIR / "scaler_X.pkl")
    print(f"  scaler_X 已保存")

    # 保存训练集末尾上下文（推理时与测试集拼接，保证时序连续）
    # 每个特征可能窗口不同，取最大窗口保存
    max_window = max(
        PER_FEATURE_PARAMS.get(t, {}).get('WINDOW', WINDOW)
        for t in TARGET_COLS
    )
    X_train_tail = {
        t: X_train[-(PER_FEATURE_PARAMS.get(t, {}).get('WINDOW', WINDOW)):]
        for t in TARGET_COLS
    }
    joblib.dump(X_train_tail, MODEL_DIR / "X_train_tail.pkl")
    print(f"  训练集末尾上下文已保存（最大窗口={max_window}）")

    # 3. 逐特征训练
    print("\n[3/4] 逐特征独立训练...")
    split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:split], X_train[split:]

    all_test_preds = []
    all_train_maes = []
    all_noise_maes = []

    for i, target in enumerate(TARGET_COLS):
        fp      = PER_FEATURE_PARAMS.get(target, {})
        window  = fp.get('WINDOW',       WINDOW)
        channels= fp.get('NUM_CHANNELS', NUM_CHANNELS)
        pat     = fp.get('PATIENCE',     PATIENCE)
        lr      = fp.get('LR',           LR)

        print(f"\n  ── [{i+1}/6] {target}  (window={window}) ──")

        y_col     = y_train_raw[:, i:i+1]
        scaler_y  = StandardScaler().fit(y_col[:split])
        y_tr_norm = scaler_y.transform(y_col[:split]).ravel()
        y_va_norm = scaler_y.transform(y_col[split:]).ravel()

        train_ds = TimeSeriesDataset(X_tr, y_tr_norm, window)
        val_ds   = TimeSeriesDataset(X_va, y_va_norm, window)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=USE_AMP)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=USE_AMP)
        print(f"    训练样本: {len(train_ds):,}   验证样本: {len(val_ds):,}")

        model = TCNDenoiser(
            input_size=X_train.shape[1],   # 17
            num_channels=channels,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        ).to(DEVICE)
        model = train_one(model, train_loader, val_loader, lr, pat)

        # 保存模型权重和归一化器
        torch.save(model.state_dict(), MODEL_DIR / f"model_{target}.pt")
        joblib.dump(scaler_y, MODEL_DIR / f"scaler_y_{target}.pkl")
        print(f"    模型已保存 → tcn_models/model_{target}.pt")

        # 训练集 MAE
        model.eval()
        full_ds = TimeSeriesDataset(X_train, scaler_y.transform(y_col).ravel(), window)
        full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=USE_AMP)
        preds_norm = []
        with torch.no_grad():
            for xb, _ in full_loader:
                preds_norm.append(model(xb.to(DEVICE)).cpu().numpy())
        preds_real = scaler_y.inverse_transform(
            np.concatenate(preds_norm).reshape(-1, 1)).ravel()
        true_real  = y_col[window:].ravel()
        noise_real = X_train_raw[window:, i]   # 对应 Error_* 列（前6列）

        noise_mae = mean_absolute_error(true_real, noise_real)
        tcn_mae   = mean_absolute_error(true_real, preds_real)
        improve   = (noise_mae - tcn_mae) / noise_mae * 100
        all_train_maes.append(tcn_mae)
        all_noise_maes.append(noise_mae)
        print(f"    噪声直接 MAE={noise_mae:.4f}  TCN MAE={tcn_mae:.4f}  提升{improve:+.1f}%")

        # 测试集预测
        X_ctx    = np.vstack([X_train[-window:], X_test])
        y_dummy  = np.zeros(len(X_ctx))
        test_ds  = TimeSeriesDataset(X_ctx, y_dummy, window)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=USE_AMP)
        test_norm = []
        with torch.no_grad():
            for xb, _ in test_loader:
                test_norm.append(model(xb.to(DEVICE)).cpu().numpy())
        test_real = scaler_y.inverse_transform(
            np.concatenate(test_norm).reshape(-1, 1)).ravel()
        all_test_preds.append(test_real)

    print(f"\n  训练集总体平均: 噪声={np.mean(all_noise_maes):.4f}  "
          f"TCN={np.mean(all_train_maes):.4f}  "
          f"提升{(np.mean(all_noise_maes)-np.mean(all_train_maes))/np.mean(all_noise_maes)*100:+.1f}%")

    # 4. 输出 CSV
    print(f"\n[4/4] 输出预测结果...")
    pred_matrix  = np.stack(all_test_preds, axis=1)
    pred_strings = [' '.join(map(str, row)) for row in pred_matrix]
    result_df    = pd.DataFrame({'Predicted_Value': pred_strings})
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"  已保存至 {OUTPUT_PATH}（共 {len(result_df):,} 行）")

    elapsed = time.time() - start_time
    print(f"\n总耗时：{elapsed:.2f} 秒")
    print("=" * 65)


if __name__ == '__main__':
    main()
