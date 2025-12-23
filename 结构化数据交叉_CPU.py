# coding=utf-8
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from bayes_opt import BayesianOptimization


# 贝叶斯优化XGBoost类 (与之前版本相同)
class BayesianXgboostOptimizer:
    def __init__(self, n_iter=20, random_state=42):
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_params = None
        self.best_boost_round = None
        self.best_score = np.inf

    def xgb_evaluate(self, learning_rate, max_depth, min_child_weight, gamma,
                     subsample, colsample_bytree, reg_alpha, reg_lambda, num_boost_round):
        params = {
            'learning_rate': round(learning_rate, 3),
            'max_depth': int(round(max_depth)),
            'min_child_weight': round(min_child_weight, 1),
            'gamma': round(gamma, 2),
            'subsample': round(subsample, 2),
            'colsample_bytree': round(colsample_bytree, 2),
            'reg_alpha': round(reg_alpha, 2),
            'reg_lambda': round(reg_lambda, 2),
            'seed': self.random_state,
            'objective': 'reg:squarederror',
            'nthread': -1
        }
        num_boost_round = int(round(num_boost_round))

        sample_size = min(45000, len(self.X_train))
        sample_pos = np.random.choice(len(self.X_train), sample_size, replace=False)
        X_sample = self.X_train[sample_pos]
        y_sample = self.y_train.iloc[sample_pos]

        dtrain = xgb.DMatrix(X_sample, label=y_sample)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)

        model = xgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            evals=[(dval, 'validation')], early_stopping_rounds=15, verbose_eval=False
        )
        best_iter = max(model.best_iteration, 1)
        y_pred = model.predict(dval, iteration_range=(0, best_iter))
        mae = mean_absolute_error(self.y_val, y_pred)

        if mae < self.best_score:
            self.best_score = mae
            self.best_params = params
            self.best_boost_round = best_iter
        return -mae

    def optimize(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        pbounds = {
            'learning_rate': (0.08, 0.15),
            'max_depth': (4, 7),  # 减小最大深度  (6, 8),
            'min_child_weight': (1.5, 2.5),
            'gamma': (0.2, 0.6),
            'subsample': (0.85, 0.95),
            'colsample_bytree': (0.85, 0.95),
            'reg_alpha': (0.3, 1.0),
            'reg_lambda': (0.3, 1.0),
            'num_boost_round': (150, 300)  # 减少 boosting 轮数   (250, 450)
        }
        optimizer = BayesianOptimization(
            f=self.xgb_evaluate, pbounds=pbounds, random_state=self.random_state, verbose=1
        )
        optimizer.maximize(init_points=5, n_iter=self.n_iter)
        print(f"最优MAE: {self.best_score:.4f}")
        return self.best_params, self.best_boost_round


# 增强特征工程函数（进一步优化版）
# 增强特征工程函数（优化版 - 解决 PerformanceWarning）
# 增强特征工程函数（优化版 - 解决 PerformanceWarning）
def enhance_features(data, core_columns, noise_columns):
    """
    增强特征工程函数，基于物理关联和误差信息构建特征，并使用动态滑动窗口。
    通过一次性合并列来优化性能，避免 DataFrame 碎片化警告。
    """
    df = data.copy()
    # 定义动态窗口大小
    window_sizes = {
        'large': 12,  # 用于平滑变化较慢的特征，如温度、密度
        'medium': 6,  # 用于一般特征
        'small': 3  # 用于变化较快或需要保留细节的特征，如信号强度
    }

    # 1. 误差列利用
    for err_col in noise_columns:
        df[f'{err_col}_weight'] = 1 / df[err_col].replace(0, 1e-8)
    for core_col in core_columns:
        matched_err = next((err for err in noise_columns if err == f'Error_{core_col}'), None)
        if matched_err:
            err_col = matched_err
            df[f'{core_col}_x_{err_col}'] = df[core_col] * df[err_col]
            df[f'{core_col}_div_{err_col}'] = df[core_col] / df[err_col].replace(0, 1e-8)

    # 2. 新增更多具有物理意义的交叉特征
    if 'T_SONIC' in core_columns:
        df['T_SONIC_sq'] = df['T_SONIC'] ** 2
        for col in core_columns:
            if col != 'T_SONIC':
                df[f'{col}_norm_by_T'] = df[col] / (df['T_SONIC'] + 1e-8)

    if 'CO2_density' in core_columns and 'CO2_sig_strgth' in core_columns:
        df['CO2_density_x_sig_strgth'] = df['CO2_density'] * df['CO2_sig_strgth']
    if 'H2O_density' in core_columns and 'H2O_sig_strgth' in core_columns:
        df['H2O_density_x_sig_strgth'] = df['H2O_density'] * df['H2O_sig_strgth']

    if 'CO2_density' in core_columns and 'H2O_density' in core_columns:
        df['CO2_H2O_density_ratio'] = df['CO2_density'] / df['H2O_density'].replace(0, 1e-8)
        if 'T_SONIC' in core_columns:
            df['CO2_H2O_T_ratio'] = df['CO2_H2O_density_ratio'] / (df['T_SONIC'] + 1e-8)

    for i in range(len(core_columns)):
        for j in range(i + 1, len(core_columns)):
            col1, col2 = core_columns[i], core_columns[j]
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

    # 3. 动态滑动窗口统计特征 (优化部分)
    # 创建一个字典来存储所有新生成的滑动窗口特征
    rolling_features = {}

    for col in df.columns:
        # 根据列名选择合适的窗口大小
        if 'T_SONIC' in col or '_density' in col:
            window = window_sizes['large']
        elif '_sig_strgth' in col:
            window = window_sizes['small']
        else:
            window = window_sizes['medium']

        # 将计算出的特征存入字典，而不是直接插入原DataFrame
        rolling_features[f'{col}_mean'] = df[col].rolling(window=window, min_periods=1).mean()
        rolling_features[f'{col}_std'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
        rolling_features[f'{col}_max'] = df[col].rolling(window=window, min_periods=1).max()
        rolling_features[f'{col}_min'] = df[col].rolling(window=window, min_periods=1).min()

    # 4. 一次性将所有滑动窗口特征合并到原始 DataFrame 中
    # 将字典转换为 DataFrame，然后使用 pd.concat 进行合并
    rolling_df = pd.DataFrame(rolling_features, index=df.index)
    df = pd.concat([df, rolling_df], axis=1)

    # 填充缺失值
    df = df.bfill().ffill()
    return df


def remove_outliers(df, columns, z_threshold=3.5):
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[columns]))
    mask = (z_scores < z_threshold).all(axis=1)
    filtered_df = df[mask]
    if len(filtered_df) < 1000:
        new_threshold = z_threshold + 0.5
        print(f"样本量不足，放宽异常值阈值至{new_threshold}")
        return remove_outliers(df, columns, new_threshold)
    return filtered_df


# 主程序
if __name__ == "__main__":
    start_time = time.time()

    columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr',
               'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
    noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
                     'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
    use_cols = columns + noise_columns

    train_dataSet = pd.read_csv(
        r'data/加噪数据集/modified_数据集Time_Series661_detail.dat',
        usecols=use_cols
    )
    test_dataSet = pd.read_csv(
        r'data/加噪数据集/modified_数据集Time_Series662_detail.dat',
        usecols=use_cols
    )

    print("处理异常值前训练集大小：", len(train_dataSet))
    train_dataSet = remove_outliers(train_dataSet, columns)
    print("处理异常值后训练集大小：", len(train_dataSet))

    print("增强特征前维度（原始核心列+误差列）：", len(use_cols))
    X_train_raw = enhance_features(train_dataSet, columns, noise_columns)
    X_test_raw = enhance_features(test_dataSet, columns, noise_columns)
    print("增强特征后维度：", X_train_raw.shape[1])

    y = train_dataSet[columns]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y, test_size=0.15, random_state=42
    )
    X_test = X_test_raw
    y_test = test_dataSet[columns]

    scaler_X = RobustScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    all_best_params = []
    all_best_boost_rounds = []
    print("\n开始贝叶斯优化 (6个独立模型, 优化目标: MAE)...")
    optimizer = BayesianXgboostOptimizer(n_iter=20, random_state=42)

    for i, target_col in enumerate(columns):
        print(f"\n--- 正在优化目标: {target_col} ({i + 1}/6) ---")
        y_train_col, y_val_col = y_train.iloc[:, i], y_val.iloc[:, i]
        best_params, best_boost_round = optimizer.optimize(X_train, y_train_col, X_val, y_val_col)
        print(f"'{target_col}' 的最优参数:", best_params)
        print(f"'{target_col}' 的最优迭代次数:", best_boost_round)
        all_best_params.append(best_params)
        all_best_boost_rounds.append(best_boost_round)

    print("\n--- 正在训练 6 个最终模型并预测 ---")
    dtest = xgb.DMatrix(X_test)
    all_predictions_list = []
    for i, target_col in enumerate(columns):
        print(f"训练和预测: {target_col} ({i + 1}/6)")
        y_train_col, y_val_col = y_train.iloc[:, i], y_val.iloc[:, i]
        dtrain_final = xgb.DMatrix(X_train, label=y_train_col)
        dval_final = xgb.DMatrix(X_val, label=y_val_col)
        params, boost_round = all_best_params[i], all_best_boost_rounds[i]

        model = xgb.train(
            params, dtrain_final, num_boost_round=boost_round,
            evals=[(dval_final, 'validation')], early_stopping_rounds=15, verbose_eval=False
        )
        best_iter = max(model.best_iteration, 1)
        y_predict_col = model.predict(dtest, iteration_range=(0, best_iter))
        all_predictions_list.append(y_predict_col)

    y_predict = np.stack(all_predictions_list, axis=1)
    results = []
    for true, pred in zip(y_test.values, y_predict):
        error = np.abs(true - pred)
        results.append([
            ' '.join(f"{x:.5f}" for x in true),
            ' '.join(f"{x:.5f}" for x in pred),
            ' '.join(f"{x:.5f}" for x in error)
        ])
    result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
    result_df.to_csv("result_Bayesian_优化版_v2.csv", index=False)

    error_cols = result_df['Error'].str.split(' ', expand=True).apply(pd.to_numeric)
    print("\n6个变量的平均绝对误差 (基于加噪的y_test)：\n", error_cols.mean())
    overall_mean_error = error_cols.mean().mean()
    print(f"总体平均误差 (基于加噪的y_test): {overall_mean_error:.4f}")
    print(f"\n总耗时：{time.time() - start_time:.3f}秒")