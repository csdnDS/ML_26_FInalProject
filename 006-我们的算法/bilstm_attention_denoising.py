# 时间序列传感器去噪 —— 双向双层 LSTM + 注意力机制
# 算法原理：
#   利用滑动窗口将时序数据转化为监督学习问题：
#     输入: 过去 WINDOW 步的带噪声观测序列 (Error_* 列)
#     输出: 当前步的真实传感器值 (TARGET_COLS)
#   双向 LSTM 同时利用前向和后向时序上下文（去噪任务有完整序列，可利用未来信息）
#   注意力机制对窗口内所有时间步加权求和，自动聚焦最重要的时刻，优于直接取末位隐状态

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

# ─── 超参数 ────────────────────────────────────────────────────────────────────
WINDOW      = 60     # 30→60：更长窗口，去噪任务中更多上下文有助于判断噪声
HIDDEN_SIZE = 128    # LSTM 隐层维度
NUM_LAYERS  = 2      # LSTM 层数
DROPOUT     = 0.3    # 0.2→0.3：加强正则化，抑制过拟合（Train/Val MSE差距过大）
BATCH_SIZE  = 1024   # 256→1024：更大批次，GPU利用率更高，梯度更稳定
EPOCHS      = 200    # 150→200：给模型更多收敛空间
LR          = 5e-4   # 1e-3→5e-4：更小初始学习率，收敛更稳定
PATIENCE    = 25     # 15→25：更宽松早停，避免过早中断
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = torch.cuda.is_available()  # 混合精度训练（RTX支持，速度提升约1.5x）

# ─── 数据路径（相对于本脚本所在目录的上级项目根目录）────────────────────────
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
    """将归一化后的序列切成 (window, 6) → (6,) 的样本对"""

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int):
        # X: (T, 6)  y: (T, 6)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.window]          # (window, 6)
        y_val = self.y[idx + self.window]                # (6,)
        return x_seq, y_val


# ─── 注意力模块 ────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    """对 LSTM 输出的所有时间步做加权求和"""
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, window, hidden_size)
        weights = torch.softmax(self.score(lstm_out), dim=1)  # (batch, window, 1)
        context = (weights * lstm_out).sum(dim=1)              # (batch, hidden_size)
        return context


# ─── 双向 LSTM + 注意力模型 ────────────────────────────────────────────────────
class BiLSTMAttentionDenoiser(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,                          # 双向：同时利用前后文
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_size * 2)      # 双向输出维度翻倍
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # x: (batch, window, input_size)
        out, _ = self.lstm(x)           # (batch, window, hidden*2)
        context = self.attention(out)   # (batch, hidden*2)
        return self.head(context)       # (batch, output_size)


# ─── 训练函数 ──────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=7, factor=0.5
    )
    criterion = nn.MSELoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=USE_AMP)  # 混合精度缩放器

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    print(f"\n  {'Epoch':<8} {'Train MSE':<16} {'Val MSE':<16} {'LR'}")
    print("  " + "-" * 58)

    for epoch in range(1, EPOCHS + 1):
        # ── 训练 ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # ── 验证 ──
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
            print(f"  {epoch:<8} {train_loss:<16.6f} {val_loss:<16.6f} {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  早停触发 (epoch={epoch}，最佳 Val MSE={best_val_loss:.6f})")
                break

    model.load_state_dict(best_state)
    return model


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  双向 LSTM + 注意力机制 时序去噪算法")
    print(f"  设备: {DEVICE}  窗口: {WINDOW}  隐层: {HIDDEN_SIZE}×2(双向)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据集...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  训练集: {len(train_df):,} 行   测试集: {len(test_df):,} 行")

    X_train_raw = train_df[NOISE_COLS].values.astype(np.float64)
    y_train_raw = train_df[TARGET_COLS].values.astype(np.float64)
    X_test_raw  = test_df[NOISE_COLS].values.astype(np.float64)

    # 2. 归一化（只在训练集上 fit）
    print("\n[2/5] 特征归一化 (StandardScaler)...")
    scaler_X = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train_raw)

    X_train = scaler_X.transform(X_train_raw)
    y_train = scaler_y.transform(y_train_raw)
    X_test  = scaler_X.transform(X_test_raw)

    # 3. 构建数据集：训练 80% / 验证 20%
    print("\n[3/5] 构建训练/验证集...")
    split    = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train[:split],  y_train[:split]
    X_va, y_va = X_train[split:],  y_train[split:]

    train_ds = TimeSeriesDataset(X_tr, y_tr, WINDOW)
    val_ds   = TimeSeriesDataset(X_va, y_va, WINDOW)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=USE_AMP)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=USE_AMP)
    print(f"  训练样本: {len(train_ds):,}   验证样本: {len(val_ds):,}")

    # 4. 训练模型
    print("\n[4/5] 训练 双向LSTM+注意力 模型...")
    model = BiLSTMAttentionDenoiser(
        input_size=len(NOISE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(TARGET_COLS),
        dropout=DROPOUT,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {total_params:,}")
    model = train_model(model, train_loader, val_loader)

    # 4b. 在训练集上报告 MAE（与真实值对比）
    print("\n  训练集验证（与真实值对比）...")
    model.eval()
    all_preds = []
    ds_full = TimeSeriesDataset(X_train, y_train, WINDOW)
    loader_full = DataLoader(ds_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=USE_AMP)
    with torch.no_grad():
        for xb, _ in loader_full:
            pred = model(xb.to(DEVICE)).cpu().numpy()
            all_preds.append(pred)
    preds_norm  = np.vstack(all_preds)
    preds_real  = scaler_y.inverse_transform(preds_norm)
    true_real   = y_train_raw[WINDOW:]   # 对应窗口偏移

    # 噪声基线（直接用噪声列作为预测）
    noise_baseline = X_train_raw[WINDOW:]
    print(f"  {'通道':<30} {'噪声直接 MAE':<22} {'BiLSTM+Attn MAE':<20} {'提升'}")
    maes_base, maes_lstm = [], []
    for i, col in enumerate(TARGET_COLS):
        base_mae = mean_absolute_error(true_real[:, i], noise_baseline[:, i])
        lstm_mae = mean_absolute_error(true_real[:, i], preds_real[:, i])
        improve  = (base_mae - lstm_mae) / base_mae * 100
        maes_base.append(base_mae)
        maes_lstm.append(lstm_mae)
        print(f"  {col:<30} {base_mae:<22.4f} {lstm_mae:<20.4f} {improve:+.1f}%")
    print(f"\n  {'总体平均':<30} {np.mean(maes_base):<22.4f} "
          f"{np.mean(maes_lstm):<20.4f} "
          f"{(np.mean(maes_base)-np.mean(maes_lstm))/np.mean(maes_base)*100:+.1f}%")

    # 5. 对测试集预测
    print("\n[5/5] 对测试集（662）进行预测，输出 CSV...")
    # 拼接训练集末尾 WINDOW 行 + 测试集，形成连续序列供滑窗使用
    X_context = np.vstack([X_train[-WINDOW:], X_test])
    # 对齐 y（测试集无真实值，用 0 占位）
    y_dummy   = np.zeros((len(X_context), len(TARGET_COLS)))
    test_full_ds = TimeSeriesDataset(X_context, y_dummy, WINDOW)
    test_loader  = DataLoader(test_full_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=USE_AMP)

    test_preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in test_loader:
            pred = model(xb.to(DEVICE)).cpu().numpy()
            test_preds.append(pred)
    test_preds_norm = np.vstack(test_preds)
    test_preds_real = scaler_y.inverse_transform(test_preds_norm)

    pred_strings = [' '.join(map(str, row)) for row in test_preds_real]
    result_df = pd.DataFrame({'Predicted_Value': pred_strings})
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"  预测结果已保存至 {OUTPUT_PATH}（共 {len(result_df):,} 行）")

    elapsed = time.time() - start_time
    print(f"\n总耗时：{elapsed:.2f} 秒")
    print("=" * 60)


if __name__ == '__main__':
    main()
