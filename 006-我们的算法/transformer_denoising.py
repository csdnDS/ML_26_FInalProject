# 时间序列传感器去噪 —— Transformer 编码器（每特征独立训练）
#
# 算法原理：
#   Transformer 用多头自注意力（Multi-Head Self-Attention）对时序建模：
#     - 全局注意力：窗口内任意两个时间步可以直接交互，不受距离限制
#     - 优于 TCN/LSTM：TCN 靠膨胀卷积逐层传递，Transformer 一步直达
#     - 位置编码：sin/cos 编码保留时序顺序信息
#     - 取最后时间步的输出接 MLP head，预测当前时刻的去噪值
#
#   输入特征（10维）：
#     · 6维噪声列（Error_*）
#     · 4维时间周期编码（hour_sin/cos 日变化、doy_sin/cos 季节变化）
#
#   参考论文：Vaswani et al., "Attention Is All You Need", NeurIPS 2017

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
WINDOW        = 128      # 滑动窗口（比 TCN 更大，充分利用全局注意力）
D_MODEL       = 128      # Transformer 嵌入维度
NHEAD         = 8        # 注意力头数（D_MODEL 必须整除 NHEAD）
NUM_LAYERS    = 3        # Transformer 编码器层数
DIM_FF        = 256      # 前馈网络隐层维度
DROPOUT       = 0.1
BATCH_SIZE    = 512      # Transformer 显存占用更高，适当减小
EPOCHS        = 200
LR            = 1e-4     # Transformer 通常用更小的学习率
PATIENCE      = 20
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP       = torch.cuda.is_available()

# ─── 每特征独立超参数 ──────────────────────────────────────────────────────────
PER_FEATURE_PARAMS = {
    'CO2_density':           {'WINDOW': 256, 'PATIENCE': 25},
    'CO2_density_fast_tmpr': {'WINDOW': 256, 'PATIENCE': 25},
}

# ─── 数据路径 ──────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
TRAIN_PATH  = str(_ROOT / "001-数据集/ML期末数据集（含真实值）/modified_数据集Time_Series661.dat")
TEST_PATH   = str(_ROOT / "001-数据集/ML期末数据集（不含真实值）/modified_数据集Time_Series662.dat")
OUTPUT_PATH = str(Path(__file__).resolve().parent / "result_Transformer.csv")
MODEL_DIR   = Path(__file__).resolve().parent / "transformer_models"

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
# 特征工程（与 tcn_denoising.py 相同）
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> np.ndarray:
    """噪声列 + 时间周期编码，共 10 维"""
    base = df[NOISE_COLS].values.astype(np.float64)
    ts   = pd.to_datetime(df['TIMESTAMP'], format='mixed')
    hour = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    doy  = ts.dt.day_of_year.astype(float)
    time_feats = np.stack([
        np.sin(2 * np.pi * hour / 24).values,
        np.cos(2 * np.pi * hour / 24).values,
        np.sin(2 * np.pi * doy  / 365).values,
        np.cos(2 * np.pi * doy  / 365).values,
    ], axis=1)
    return np.hstack([base, time_feats])   # (n, 10)


# ══════════════════════════════════════════════════════════════════════════════
# Transformer 网络结构
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """经典 sin/cos 位置编码，让模型知道序列中每个位置的顺序"""
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerDenoiser(nn.Module):
    """
    Transformer 编码器去噪模型：
      输入: (batch, window, n_features)
      输出: (batch,) 单特征去噪值
    """
    def __init__(self, input_size, d_model, nhead, num_layers, dim_ff, dropout):
        super().__init__()
        # 输入投影：将原始特征映射到 d_model 维
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN，训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)       # → (batch, seq_len, d_model)
        x = self.pos_enc(x)          # 加位置编码
        x = self.encoder(x)          # Transformer 编码
        x = x[:, -1, :]             # 取最后时间步
        return self.head(x).squeeze(-1)


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
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=lr * 0.01)
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
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"    {epoch:<8} {train_loss:<16.6f} {val_loss:<16.6f} "
                  f"{optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    早停 (epoch={epoch}, best_val={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    print("=" * 65)
    print("  Transformer 编码器  【每特征独立训练 · 全局自注意力】")
    print(f"  设备: {DEVICE}   d_model={D_MODEL}   heads={NHEAD}   layers={NUM_LAYERS}")
    print(f"  全局窗口: {WINDOW}   CO2窗口: {PER_FEATURE_PARAMS['CO2_density']['WINDOW']}")
    print(f"  输入维度: 10  (噪声6 + 时间编码4)")
    print("=" * 65)

    # 1. 加载数据并构造特征
    print("\n[1/4] 加载数据集并构造输入特征...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  训练集: {len(train_df):,} 行   测试集: {len(test_df):,} 行")

    X_train_raw = build_features(train_df)
    X_test_raw  = build_features(test_df)
    y_train_raw = train_df[TARGET_COLS].values.astype(np.float64)
    print(f"  特征矩阵: {X_train_raw.shape}")

    # 2. 归一化
    print("\n[2/4] 输入特征归一化...")
    scaler_X = StandardScaler().fit(X_train_raw)
    X_train  = scaler_X.transform(X_train_raw)
    X_test   = scaler_X.transform(X_test_raw)
    joblib.dump(scaler_X, MODEL_DIR / "scaler_X.pkl")

    # 3. 逐特征训练
    print("\n[3/4] 逐特征独立训练...")
    split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:split], X_train[split:]

    all_test_preds = []
    all_train_maes = []
    all_noise_maes = []

    for i, target in enumerate(TARGET_COLS):
        fp      = PER_FEATURE_PARAMS.get(target, {})
        window  = fp.get('WINDOW',   WINDOW)
        pat     = fp.get('PATIENCE', PATIENCE)
        lr      = fp.get('LR',       LR)

        print(f"\n  ── [{i+1}/6] {target}  (window={window}) ──")

        y_col     = y_train_raw[:, i:i+1]
        scaler_y  = StandardScaler().fit(y_col[:split])
        y_tr_norm = scaler_y.transform(y_col[:split]).ravel()
        y_va_norm = scaler_y.transform(y_col[split:]).ravel()
        joblib.dump(scaler_y, MODEL_DIR / f"scaler_y_{target}.pkl")

        train_ds = TimeSeriesDataset(X_tr, y_tr_norm, window)
        val_ds   = TimeSeriesDataset(X_va, y_va_norm, window)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=USE_AMP)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=USE_AMP)
        print(f"    训练样本: {len(train_ds):,}   验证样本: {len(val_ds):,}")

        model = TransformerDenoiser(
            input_size=X_train.shape[1],
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_ff=DIM_FF,
            dropout=DROPOUT,
        ).to(DEVICE)
        model = train_one(model, train_loader, val_loader, lr, pat)
        torch.save(model.state_dict(), MODEL_DIR / f"model_{target}.pt")

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
        noise_real = X_train_raw[window:, i]

        noise_mae = mean_absolute_error(true_real, noise_real)
        tf_mae    = mean_absolute_error(true_real, preds_real)
        improve   = (noise_mae - tf_mae) / noise_mae * 100
        all_train_maes.append(tf_mae)
        all_noise_maes.append(noise_mae)
        print(f"    噪声直接 MAE={noise_mae:.4f}  Transformer MAE={tf_mae:.4f}  提升{improve:+.1f}%")

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
          f"Transformer={np.mean(all_train_maes):.4f}  "
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
