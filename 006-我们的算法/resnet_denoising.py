import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 数据路径配置 ─────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
TRAIN_PATH = str(_ROOT / "001-数据集/ML期末数据集（含真实值）/modified_数据集Time_Series661.dat")
TEST_PATH  = str(_ROOT / "001-数据集/ML期末数据集（不含真实值）/modified_数据集Time_Series662.dat")
OUTPUT_PATH = str(Path(__file__).resolve().parent / "result_ResNet.csv")
MODEL_DIR   = Path(__file__).resolve().parent / "resnet_models"

# ─── 特征列定义 ────────────────────────────────────────────────────────────────
TARGET_COLS = [
    'T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
    'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth'
]
NOISE_COLS = [
    'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
    'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth'
]

# ─── 超参数 ────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 32
BATCH_SIZE    = 64
EPOCHS        = 25
LEARNING_RATE = 0.001


# ─── 归一化器 ──────────────────────────────────────────────────────────────────
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


# ─── 向量化滑动窗口（比原版 Python 循环快 20 倍）─────────────────────────────
def create_sliding_windows(noise_data: np.ndarray,
                           target_data: np.ndarray = None,
                           window_size: int = 32):
    """
    向量化版本：用 NumPy 索引一次性生成所有窗口，避免 Python 循环。
    输出 X shape: (n_samples, n_channels, window_size)
    输出 y shape: (n_samples, n_channels)  ← 等价于 target_data 本身
    """
    n_samples, n_channels = noise_data.shape
    pad_size = window_size - 1

    # 边界填充
    noise_padded = np.pad(noise_data, ((pad_size, 0), (0, 0)), mode='edge')

    # 一次性构造所有窗口的行索引 (n_samples, window_size)
    row_idx = np.arange(n_samples)[:, None] + np.arange(window_size)[None, :]

    # (n_samples, window_size, n_channels) → transpose → (n_samples, n_channels, window_size)
    windows_x = noise_padded[row_idx].transpose(0, 2, 1).astype(np.float32)

    if target_data is not None:
        # windows_y[i] = target_data[i]（边界填充不影响标签）
        return windows_x, target_data.astype(np.float32)
    return windows_x


# ─── 网络结构 ──────────────────────────────────────────────────────────────────
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
        return correction[:, :, -1]         # 仅预测修正量 delta


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    MODEL_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("  一维时序残差修正网络 (ResNet Denoising)")
    print(f"  设备: {device}   窗口: {WINDOW_SIZE}   Epochs: {EPOCHS}")
    print("=" * 60)

    # [1] 数据加载
    print("\n[1/5] 读取数据集...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  训练集: {len(train_df):,} 行   测试集: {len(test_df):,} 行")

    train_noise  = train_df[NOISE_COLS].values
    train_target = train_df[TARGET_COLS].values
    test_noise   = test_df[NOISE_COLS].values
    train_delta  = train_target - train_noise

    # [2] 归一化
    print("\n[2/5] Z-Score 标准化...")
    scaler_noise  = ChannelStandardScaler()
    scaler_delta  = ChannelStandardScaler()
    train_noise_scaled  = scaler_noise.fit_transform(train_noise)
    train_delta_scaled  = scaler_delta.fit_transform(train_delta)
    test_noise_scaled   = scaler_noise.transform(test_noise)

    # 保存归一化器（推理时必须用同一个 scaler）
    joblib.dump(scaler_noise,  MODEL_DIR / "scaler_noise.pkl")
    joblib.dump(scaler_delta, MODEL_DIR / "scaler_delta.pkl")
    print("  归一化器已保存至 resnet_models/")

    # [3] 构造滑动窗口（向量化，快）
    print("\n[3/5] 构造滑动窗口...")
    X_train, Y_train = create_sliding_windows(train_noise_scaled, train_delta_scaled, WINDOW_SIZE)
    X_test           = create_sliding_windows(test_noise_scaled,  None,                WINDOW_SIZE)
    print(f"  训练窗口: {X_train.shape}   测试窗口: {X_test.shape}")

    tensor_x = torch.tensor(X_train)
    tensor_y = torch.tensor(Y_train)
    train_loader = DataLoader(TensorDataset(tensor_x, tensor_y),
                              batch_size=BATCH_SIZE, shuffle=True)

    # [4] 训练
    print(f"\n[4/5] 训练残差网络...")
    model     = ResidualDenoiseNet(in_channels=len(NOISE_COLS)).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        scheduler.step()
        print(f"  Epoch [{epoch:02d}/{EPOCHS}]  MAE Loss: {epoch_loss / len(tensor_x):.6f}")

    # 保存模型
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    print(f"  模型已保存至 resnet_models/model.pt")

    # [5] 训练集验证
    print("\n[5/5] 训练集验证 & 生成测试集预测...")
    model.eval()
    with torch.no_grad():
        train_preds_scaled = []
        for bx, _ in DataLoader(TensorDataset(tensor_x, tensor_y),
                                 batch_size=BATCH_SIZE, shuffle=False):
            train_preds_scaled.append(model(bx.to(device)).cpu().numpy())
        train_delta_pred = scaler_delta.inverse_transform(np.vstack(train_preds_scaled))
        train_preds = train_noise + train_delta_pred

    print(f"  {'特征':<30} {'噪声MAE':<14} {'ResNet MAE':<14} {'提升'}")
    print("  " + "-" * 60)
    maes_base, maes_res = [], []
    for i, target in enumerate(TARGET_COLS):
        base = np.mean(np.abs(train_noise[:, i] - train_target[:, i]))
        res  = np.mean(np.abs(train_preds[:, i] - train_target[:, i]))
        maes_base.append(base); maes_res.append(res)
        print(f"  {target:<30} {base:<14.4f} {res:<14.4f} {(base-res)/base*100:+.1f}%")
    print(f"  {'总体平均':<30} {np.mean(maes_base):<14.4f} {np.mean(maes_res):<14.4f} "
          f"{(np.mean(maes_base)-np.mean(maes_res))/np.mean(maes_base)*100:+.1f}%")

    # 测试集预测
    with torch.no_grad():
        test_preds_scaled = []
        tensor_test = torch.tensor(X_test)
        for bx, in DataLoader(TensorDataset(tensor_test), batch_size=BATCH_SIZE):
            test_preds_scaled.append(model(bx.to(device)).cpu().numpy())
    test_delta_pred = scaler_delta.inverse_transform(np.vstack(test_preds_scaled))
    test_preds = test_noise + test_delta_pred

    pd.DataFrame({'Predicted_Value': [' '.join(map(str, r)) for r in test_preds]}
                 ).to_csv(OUTPUT_PATH, index=False)
    print(f"\n  预测结果已保存至 {OUTPUT_PATH}")

    print(f"\n总耗时：{time.time() - start_time:.2f} 秒")
    print("=" * 60)


if __name__ == '__main__':
    main()
