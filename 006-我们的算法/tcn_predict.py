# TCN 推理脚本（考试专用）
#
# 使用方法：
#   1. 将老师发的测试集文件路径填写到 TEST_PATH
#   2. 运行本脚本，直接输出 result_TCN.csv，无需重新训练
#
# 前提：已运行过 tcn_denoising.py，tcn_models/ 目录下有保存的模型文件

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── ★ 考试时只需修改这一行：填写老师发的测试集路径 ★ ───────────────────────
TEST_PATH = r"F:\ML期末大作业_250835002022_唐雨+250835002021_李恒\001-数据集\ML期末数据集（不含真实值）\modified_数据集Time_Series662.dat"
# ─────────────────────────────────────────────────────────────────────────────

_HERE       = Path(__file__).resolve().parent
MODEL_DIR   = _HERE / "tcn_models"
OUTPUT_PATH = str(_HERE / "result_TCN.csv")

WINDOW       = 64
NUM_CHANNELS = [64, 64, 64, 64]
KERNEL_SIZE  = 4
DROPOUT      = 0.2
BATCH_SIZE   = 2048
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP      = torch.cuda.is_available()

TARGET_COLS = [
    'T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
    'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth'
]
NOISE_COLS = [
    'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
    'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth'
]
EXTRA_COLS = ['Ux', 'Uy', 'Uz', 'PA', 'TA_1_1_1', 'T_SONIC_corr', 'FW']

PER_FEATURE_PARAMS = {
    'CO2_density':           {'WINDOW': WINDOW},
    'CO2_density_fast_tmpr': {'WINDOW': WINDOW},
}


# ══════════════════════════════════════════════════════════════════════════════
# 特征构造（与训练脚本完全一致）
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> np.ndarray:
    base = df[NOISE_COLS + EXTRA_COLS].values.astype(np.float64)
    ts   = pd.to_datetime(df['TIMESTAMP'], format='mixed')
    hour = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    doy  = ts.dt.day_of_year.astype(float)
    hour_sin = np.sin(2 * np.pi * hour / 24).values
    hour_cos = np.cos(2 * np.pi * hour / 24).values
    doy_sin  = np.sin(2 * np.pi * doy  / 365).values
    doy_cos  = np.cos(2 * np.pi * doy  / 365).values
    time_feats = np.stack([hour_sin, hour_cos, doy_sin, doy_cos], axis=1)
    return np.hstack([base, time_feats])


# ══════════════════════════════════════════════════════════════════════════════
# TCN 结构（与训练脚本完全一致）
# ══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)

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
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        layers, in_ch = [], input_size
        for i, out_ch in enumerate(num_channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
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
        return self.head(out[:, :, -1]).squeeze(-1)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, window):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        return self.X[idx: idx + self.window]


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  TCN 推理脚本（无需重新训练）")
    print(f"  设备: {DEVICE}   模型目录: tcn_models/")
    print("=" * 65)

    # 检查模型文件是否存在
    missing = [t for t in TARGET_COLS
               if not (MODEL_DIR / f"model_{t}.pt").exists()]
    if missing:
        print(f"\n[错误] 以下特征的模型文件不存在，请先运行 tcn_denoising.py 完成训练：")
        for t in missing:
            print(f"  ✗ tcn_models/model_{t}.pt")
        return

    # 1. 加载测试集并构造特征
    print(f"\n[1/3] 加载测试集: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    print(f"  测试集: {len(test_df):,} 行")

    # 加载训练时保存的 scaler_X（必须和训练时完全一致）
    scaler_X = joblib.load(MODEL_DIR / "scaler_X.pkl")
    X_test_raw = build_features(test_df)
    X_test     = scaler_X.transform(X_test_raw)
    print(f"  特征矩阵: {X_test.shape}")

    # 需要训练集末尾 WINDOW 行来做上下文衔接（已在训练时一并保存）
    X_train_tail = joblib.load(MODEL_DIR / "X_train_tail.pkl")

    # 2. 逐特征加载模型并推理
    print("\n[2/3] 逐特征加载模型推理...")
    all_preds = []

    for target in TARGET_COLS:
        fp     = PER_FEATURE_PARAMS.get(target, {})
        window = fp.get('WINDOW', WINDOW)

        scaler_y = joblib.load(MODEL_DIR / f"scaler_y_{target}.pkl")

        model = TCNDenoiser(
            input_size=X_test.shape[1],
            num_channels=NUM_CHANNELS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT,
        ).to(DEVICE)
        model.load_state_dict(torch.load(
            MODEL_DIR / f"model_{target}.pt",
            map_location=DEVICE, weights_only=True))
        model.eval()

        X_ctx    = np.vstack([X_train_tail[target], X_test])
        test_ds  = TimeSeriesDataset(X_ctx, window)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)

        preds_norm = []
        with torch.no_grad():
            for xb in test_loader:
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    preds_norm.append(model(xb.to(DEVICE)).cpu().numpy())

        preds_real = scaler_y.inverse_transform(
            np.concatenate(preds_norm).reshape(-1, 1)).ravel()
        all_preds.append(preds_real)
        print(f"  ✓ {target}")

    # 3. 输出 CSV
    print(f"\n[3/3] 输出预测结果...")
    pred_matrix  = np.stack(all_preds, axis=1)
    pred_strings = [' '.join(map(str, row)) for row in pred_matrix]
    pd.DataFrame({'Predicted_Value': pred_strings}).to_csv(OUTPUT_PATH, index=False)
    print(f"  已保存至 {OUTPUT_PATH}（共 {len(pred_matrix):,} 行）")
    print("=" * 65)


if __name__ == '__main__':
    main()
