# 时间序列传感器去噪 —— 自适应 Kalman 滤波
# 算法原理：
#   对每个通道独立建立随机游走观测模型：
#     状态方程：x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
#     观测方程：z_t = x_t + v_t,       v_t ~ N(0, R)
#   Kalman 增益 K = P/(P+R) 自动平衡"跟踪信号"与"抑制噪声"
#   参数 Q、R 从训练集自动估计（最大似然），无需手动调参

import time
import numpy as np
import pandas as pd
from pathlib import Path

start_time = time.time()

# ─── 数据路径（相对于本脚本所在目录的上级项目根目录）────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH  = str(_ROOT / "001-数据集/ML期末数据集（含真实值）/modified_数据集Time_Series661.dat")
TEST_PATH   = str(_ROOT / "001-数据集/ML期末数据集（不含真实值）/modified_数据集Time_Series662.dat")
OUTPUT_PATH = str(Path(__file__).resolve().parent / "result_Kalman.csv")

# ─── 特征列定义 ────────────────────────────────────────────────────────────────
TARGET_COLS = [
    'T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
    'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth'
]
NOISE_COLS = [
    'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
    'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth'
]


def kalman_filter_1d(observations: np.ndarray, Q: float, R: float) -> np.ndarray:
    """
    单通道 1D Kalman 滤波（随机游走模型）。
    observations : 带噪声的观测序列
    Q            : 过程噪声方差（信号自身变化的快慢）
    R            : 观测噪声方差（传感器噪声的强度）
    返回: 滤波后的去噪估计序列
    """
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
    """
    从训练集估计每个通道的过程噪声方差 Q 和观测噪声方差 R。
    R = Var(噪声列 - 真实列)  ← 直接量化传感器噪声
    Q = Var(真实列的一阶差分)  ← 量化信号自身变化速度
    """
    params = {}
    for target, noise in zip(TARGET_COLS, NOISE_COLS):
        true_vals  = train_df[target].values
        noisy_vals = train_df[noise].values
        R = float(np.var(noisy_vals - true_vals))
        Q = float(np.var(np.diff(true_vals)))
        params[target] = (Q, R)
    return params


def main():
    print("=" * 55)
    print("  自适应 Kalman 滤波去噪算法")
    print("=" * 55)

    # 1. 加载数据
    print("\n[1/4] 加载数据集...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  训练集: {len(train_df):,} 行   测试集: {len(test_df):,} 行")

    # 2. 从训练集估计噪声参数
    print("\n[2/4] 自动估计每个通道的 Q、R 参数...")
    params = estimate_params(train_df)
    print(f"  {'通道':<30} {'Q (信号方差)':<18} {'R (噪声方差)':<18} {'K_稳态 (增益)'}")
    for target, (Q, R) in params.items():
        K_steady = Q / (Q + R) if (Q + R) > 0 else 0
        print(f"  {target:<30} {Q:<18.6f} {R:<18.6f} {K_steady:.4f}")

    # 3. 在训练集上验证（与真实值对比）
    print("\n[3/4] 在训练集上验证效果（与真实值对比）...")
    print(f"  {'通道':<30} {'直接用噪声列 MAE':<22} {'Kalman MAE':<16} {'提升'}")
    train_maes_base, train_maes_kalman = [], []
    for target, noise in zip(TARGET_COLS, NOISE_COLS):
        true_vals  = train_df[target].values
        noisy_vals = train_df[noise].values
        Q, R = params[target]
        base_mae   = np.mean(np.abs(noisy_vals - true_vals))
        kalman_est = kalman_filter_1d(noisy_vals, Q, R)
        kalman_mae = np.mean(np.abs(kalman_est - true_vals))
        improve    = (base_mae - kalman_mae) / base_mae * 100
        train_maes_base.append(base_mae)
        train_maes_kalman.append(kalman_mae)
        print(f"  {target:<30} {base_mae:<22.4f} {kalman_mae:<16.4f} {improve:+.1f}%")

    print(f"\n  {'总体平均':<30} {np.mean(train_maes_base):<22.4f} "
          f"{np.mean(train_maes_kalman):<16.4f} "
          f"{(np.mean(train_maes_base)-np.mean(train_maes_kalman))/np.mean(train_maes_base)*100:+.1f}%")

    # 4. 对测试集预测并输出 CSV
    print("\n[4/4] 对测试集（662）进行预测，输出 CSV...")
    predictions = []
    for target, noise in zip(TARGET_COLS, NOISE_COLS):
        noisy_vals = test_df[noise].values
        Q, R = params[target]
        kalman_est = kalman_filter_1d(noisy_vals, Q, R)
        predictions.append(kalman_est)

    pred_matrix  = np.stack(predictions, axis=1)
    pred_strings = [' '.join(map(str, row)) for row in pred_matrix]
    result_df    = pd.DataFrame({'Predicted_Value': pred_strings})
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"  预测结果已保存至 {OUTPUT_PATH}")

    elapsed = time.time() - start_time
    print(f"\n总耗时：{elapsed:.2f} 秒")
    print("=" * 55)


if __name__ == '__main__':
    main()
