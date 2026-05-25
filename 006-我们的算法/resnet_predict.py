"""
ResNet 去噪推理脚本（考试专用）
────────────────────────────────
使用方法：
  1. 把老师发的测试集路径填到 TEST_PATH
  2. 直接运行：python resnet_predict.py
  3. 预测结果自动保存到 result_ResNet.csv

前提：resnet_denoising.py 已经训练过一次，
      resnet_models/ 文件夹中存有 model.pt / scaler_noise.pkl / scaler_delta.pkl
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── 修改这里 ──────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent
TEST_PATH = str(_ROOT / "001-数据集/ML期末数据集（不含真实值）/modified_数据集Time_Series662.dat")
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_PATH = str(Path(__file__).resolve().parent / "result_ResNet.csv")
MODEL_DIR   = Path(__file__).resolve().parent / "resnet_models"

NOISE_COLS = [
    'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
    'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth'
]
TARGET_COLS = [
    'T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
    'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth'
]

WINDOW_SIZE = 32
BATCH_SIZE  = 256
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── 归一化器（与训练脚本保持一致，兼容 joblib 反序列化）──────────────────────
class ChannelStandardScaler:
    def __init__(self):
        self.means = None
        self.stds  = None

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.means = data.mean(axis=0)
        self.stds  = data.std(axis=0)
        self.stds[self.stds == 0] += 1e-8
        return (data - self.means) / self.stds

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.means) / self.stds

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.stds + self.means


# ─── 网络结构（与训练时完全一致）─────────────────────────────────────────────
class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        return x * self.fc(x).view(b, c, 1)


class ResidualDenoiseNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.init_layer = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.res_block1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64)
        )
        self.se    = SqueezeExcitation1D(64)
        self.relu  = nn.ReLU()
        self.final = nn.Conv1d(64, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        feat       = self.init_layer(x)
        res_feat   = self.se(self.res_block1(feat))
        feat       = self.relu(feat + res_feat)
        correction = self.final(feat)
        return correction[:, :, -1]


# ─── 向量化滑动窗口 ────────────────────────────────────────────────────────────
def create_sliding_windows(noise_data: np.ndarray, window_size: int = 32) -> np.ndarray:
    n_samples, n_channels = noise_data.shape
    pad_size = window_size - 1
    noise_padded = np.pad(noise_data, ((pad_size, 0), (0, 0)), mode='edge')
    row_idx = np.arange(n_samples)[:, None] + np.arange(window_size)[None, :]
    return noise_padded[row_idx].transpose(0, 2, 1).astype(np.float32)


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ResNet 去噪推理（无需训练）")
    print(f"  设备: {device}   窗口: {WINDOW_SIZE}")
    print("=" * 60)

    # 检查模型文件
    for fname in ["model.pt", "scaler_noise.pkl", "scaler_delta.pkl"]:
        if not (MODEL_DIR / fname).exists():
            raise FileNotFoundError(
                f"找不到 {MODEL_DIR / fname}，请先运行 resnet_denoising.py 训练一次！"
            )

    # [1] 加载 scaler 和模型
    print("\n[1/3] 加载 scaler 和模型...")
    scaler_noise  = joblib.load(MODEL_DIR / "scaler_noise.pkl")
    scaler_delta  = joblib.load(MODEL_DIR / "scaler_delta.pkl")

    model = ResidualDenoiseNet(in_channels=len(NOISE_COLS)).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
    model.eval()
    print("  加载完成")

    # [2] 读取并预处理测试集
    print("\n[2/3] 读取测试集并构造滑动窗口...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"  测试集: {len(test_df):,} 行")

    test_noise        = test_df[NOISE_COLS].values
    test_noise_scaled = scaler_noise.transform(test_noise)
    X_test            = create_sliding_windows(test_noise_scaled, WINDOW_SIZE)
    print(f"  测试窗口: {X_test.shape}")

    # [3] 推理
    print("\n[3/3] 推理中...")
    tensor_test = torch.tensor(X_test)
    preds_scaled = []
    with torch.no_grad():
        for (bx,) in DataLoader(TensorDataset(tensor_test), batch_size=BATCH_SIZE):
            preds_scaled.append(model(bx.to(device)).cpu().numpy())
    deltas = scaler_delta.inverse_transform(np.vstack(preds_scaled))
    preds = test_noise + deltas

    pd.DataFrame({'Predicted_Value': [' '.join(map(str, r)) for r in preds]}
                 ).to_csv(OUTPUT_PATH, index=False)
    print(f"\n  预测结果已保存至 {OUTPUT_PATH}")
    print(f"  共 {len(preds):,} 行，每行 {preds.shape[1]} 个特征值")
    print("=" * 60)


if __name__ == '__main__':
    main()
