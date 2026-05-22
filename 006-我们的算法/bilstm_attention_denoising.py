# 时间序列传感器去噪 —— 双向双层 LSTM + 注意力机制（每特征独立训练）
# 算法原理：
#   利用滑动窗口将时序数据转化为监督学习问题：
#     输入: 过去 WINDOW 步的带噪声观测序列 (Error_* 列，共6通道)
#     输出: 当前步某一通道的真实传感器值
#   每个目标特征训练一个独立模型，避免不同量纲特征相互干扰
#   双向 LSTM 同时利用前向和后向时序上下文
#   注意力机制对窗口内所有时间步加权求和，自动聚焦最重要的时刻

import time
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
WINDOW      = 30     # 滑动窗口长度
HIDDEN_SIZE = 128    # LSTM 隐层维度
NUM_LAYERS  = 2      # LSTM 层数
DROPOUT     = 0.3    # Dropout 比例
BATCH_SIZE  = 1024   # 批大小
EPOCHS      = 200    # 最大训练轮数
LR          = 5e-4   # 初始学习率
PATIENCE    = 20     # 早停耐心步数
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = torch.cuda.is_available()  # 混合精度训练

# ─── 每特征独立超参数（可针对性调整）─────────────────────────────────────────
# key: 特征名, value: 覆盖全局超参数的字典（不写则用全局值）
PER_FEATURE_PARAMS = {
    'CO2_density':           {'WINDOW': 20, 'HIDDEN_SIZE': 192, 'PATIENCE': 25},
    'CO2_density_fast_tmpr': {'WINDOW': 15, 'HIDDEN_SIZE': 192, 'PATIENCE': 25},
}

# ─── 数据路径 ──────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
TRAIN_PATH  = str(_ROOT / "001-数据集/ML期末数据集（含真实值）/modified_数据集Time_Series661.dat")
TEST_PATH   = str(_ROOT / "001-数据集/ML期末数据集（不含真实值）/modified_数据集Time_Series662.dat")
OUTPUT_PATH = str(Path(__file__).resolve().parent / "result_BiLSTM_Attn.csv")

# ─── 特征列 ────────────────────────────────────────────────────────────────────
TARGET_COLS = [
    'T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
    'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth'
]
NOISE_COLS = [
    'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
    'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth'
]


# ─── 数据集类 ──────────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    """滑动窗口数据集：输入 (window, n_noise)，输出 (1,) 单特征"""

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.window]       # (window, n_noise)
        y_val = self.y[idx + self.window]             # (1,)
        return x_seq, y_val


# ─── 注意力模块 ────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.score(lstm_out), dim=1)
        return (weights * lstm_out).sum(dim=1)


# ─── 双向 LSTM + 注意力模型 ────────────────────────────────────────────────────
class BiLSTMAttentionDenoiser(nn.Module):
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
            nn.Linear(hidden_size, 1),   # 单特征输出
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        context = self.attention(out)
        return self.head(context).squeeze(-1)   # (batch,)


# ─── 单特征训练函数 ────────────────────────────────────────────────────────────
def train_one(model, train_loader, val_loader, lr, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=7, factor=0.5
    )
    criterion = nn.MSELoss()
    amp_scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    print(f"    {'Epoch':<8} {'Train MSE':<16} {'Val MSE':<16} {'LR'}")
    print("    " + "-" * 54)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                pred = model(xb)
                loss = criterion(pred, yb)
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
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == 1:
            print(f"    {epoch:<8} {train_loss:<16.6f} {val_loss:<16.6f} {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    早停触发 (epoch={epoch}，最佳 Val MSE={best_val_loss:.6f})")
                break

    model.load_state_dict(best_state)
    return model


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  双向 LSTM + 注意力机制  【每特征独立训练】")
    print(f"  设备: {DEVICE}   全局窗口: {WINDOW}   隐层: {HIDDEN_SIZE}×2(双向)")
    print("=" * 65)

    # 1. 加载数据
    print("\n[1/4] 加载数据集...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  训练集: {len(train_df):,} 行   测试集: {len(test_df):,} 行")

    X_train_raw = train_df[NOISE_COLS].values.astype(np.float64)
    y_train_raw = train_df[TARGET_COLS].values.astype(np.float64)
    X_test_raw  = test_df[NOISE_COLS].values.astype(np.float64)

    # 2. 对噪声输入统一归一化
    print("\n[2/4] 输入特征归一化 (StandardScaler)...")
    scaler_X = StandardScaler().fit(X_train_raw)
    X_train  = scaler_X.transform(X_train_raw)
    X_test   = scaler_X.transform(X_test_raw)

    # 3. 逐特征训练
    print("\n[3/4] 逐特征独立训练...")
    split = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:split], X_train[split:]

    all_test_preds  = []   # 每特征的测试集预测（原始尺度）
    all_train_maes  = []
    all_noise_maes  = []

    for i, target in enumerate(TARGET_COLS):
        params  = PER_FEATURE_PARAMS.get(target, {})
        window  = params.get('WINDOW',      WINDOW)
        hidden  = params.get('HIDDEN_SIZE', HIDDEN_SIZE)
        pat     = params.get('PATIENCE',    PATIENCE)
        lr      = params.get('LR',          LR)

        print(f"\n  ── [{i+1}/6] {target}  (window={window}, hidden={hidden}) ──")

        # 目标列单独归一化
        y_col_train = y_train_raw[:, i:i+1]
        scaler_y    = StandardScaler().fit(y_col_train[:split])
        y_tr_norm   = scaler_y.transform(y_col_train[:split]).ravel()
        y_va_norm   = scaler_y.transform(y_col_train[split:]).ravel()

        # 数据集
        train_ds = TimeSeriesDataset(X_tr, y_tr_norm, window)
        val_ds   = TimeSeriesDataset(X_va, y_va_norm, window)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=USE_AMP)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=USE_AMP)
        print(f"    训练样本: {len(train_ds):,}   验证样本: {len(val_ds):,}")

        # 模型
        model = BiLSTMAttentionDenoiser(
            input_size=len(NOISE_COLS),
            hidden_size=hidden,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)
        model = train_one(model, train_loader, val_loader, lr, pat)

        # 训练集 MAE 验证
        model.eval()
        full_ds     = TimeSeriesDataset(X_train, y_tr_norm if False else
                                        scaler_y.transform(y_col_train).ravel(), window)
        full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=USE_AMP)
        preds_norm = []
        with torch.no_grad():
            for xb, _ in full_loader:
                preds_norm.append(model(xb.to(DEVICE)).cpu().numpy())
        preds_real = scaler_y.inverse_transform(
            np.concatenate(preds_norm).reshape(-1, 1)).ravel()
        true_real  = y_col_train[window:].ravel()
        noise_real = X_train_raw[window:, i]

        noise_mae = mean_absolute_error(true_real, noise_real)
        lstm_mae  = mean_absolute_error(true_real, preds_real)
        improve   = (noise_mae - lstm_mae) / noise_mae * 100
        all_train_maes.append(lstm_mae)
        all_noise_maes.append(noise_mae)
        print(f"    噪声直接 MAE={noise_mae:.4f}  BiLSTM MAE={lstm_mae:.4f}  提升{improve:+.1f}%")

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

    # 汇总
    print(f"\n  训练集总体平均: 噪声={np.mean(all_noise_maes):.4f}  "
          f"BiLSTM={np.mean(all_train_maes):.4f}  "
          f"提升{(np.mean(all_noise_maes)-np.mean(all_train_maes))/np.mean(all_noise_maes)*100:+.1f}%")

    # 4. 输出 CSV
    print(f"\n[4/4] 输出预测结果...")
    pred_matrix = np.stack(all_test_preds, axis=1)   # (n_test, 6)
    pred_strings = [' '.join(map(str, row)) for row in pred_matrix]
    result_df = pd.DataFrame({'Predicted_Value': pred_strings})
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"  已保存至 {OUTPUT_PATH}（共 {len(result_df):,} 行）")

    elapsed = time.time() - start_time
    print(f"\n总耗时：{elapsed:.2f} 秒")
    print("=" * 65)


if __name__ == '__main__':
    main()
