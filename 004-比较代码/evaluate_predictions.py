import pandas as pd
import numpy as np

tag = True
if tag:
    pred_file_path = r'../002-基准算法/result_XGB.csv'  #基准算法结果
else:
    pred_file_path = r'../result_Kalman.csv'  #我们的算法结果
true_file_path = r"../001-数据集/ML期末数据集（含真实值）/modified_数据集Time_Series662.dat"

pred_data = pd.read_csv(pred_file_path)
true_data = pd.read_csv(true_file_path)

target_columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
true_values = true_data[target_columns].values

pred_values = np.array(pred_data['Predicted_Value'].apply(lambda x: list(map(float, x.split()))).tolist())

assert len(pred_values) == len(true_values), "预测值和真实值的行数不匹配，请检查数据"

errors = np.abs(pred_values - true_values)
mean_errors = np.mean(errors, axis=0)
overall_mean_error = np.mean(errors)

print("每个特征的平均误差：")
for feature, error in zip(target_columns, mean_errors):
    print(f"{feature}: {error:.4f}")

if tag:
    print(f"\nXGB总体平均误差: {overall_mean_error:.4f}")
else:
    print(f"\nKalman总体平均误差: {overall_mean_error:.4f}")