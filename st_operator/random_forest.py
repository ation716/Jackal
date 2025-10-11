import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


class AdaptiveCPTPredictor:
    """
    自适应CPT启发式证券预测器
    结合成本分布、市场情绪和在线学习机制
    """

    def __init__(self, lookback_days=180):
        self.lookback_days = lookback_days
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.prediction_history = []
        self.actual_history = []

    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        # 价格动量指标
        df['price_change'] = df['收盘价'].pct_change()
        df['momentum_5'] = df['收盘价'].pct_change(5)
        df['momentum_10'] = df['收盘价'].pct_change(10)

        # 波动率指标
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_10'] = df['price_change'].rolling(10).std()

        # 价格位置指标
        df['high_low_ratio'] = (df['最高价'] - df['最低价']) / df['收盘价']
        df['close_position'] = (df['收盘价'] - df['最低价']) / (df['最高价'] - df['最低价'] + 1e-8)

        # 成交量指标
        df['volume_ma_5'] = df['成交量'].rolling(5).mean()
        df['volume_ratio'] = df['成交量'] / df['volume_ma_5']

        return df

    def calculate_cost_pressure_features(self, df):
        """计算成本压力特征"""
        # 当前价格相对于各成本分位的位置
        cost_levels = ['5分位成本', '15分位成本', '50分位成本', '85分位成本', '95分位成本']

        for level in cost_levels:
            df[f'price_to_{level}'] = df['收盘价'] / df[level]
            df[f'above_{level}'] = (df['收盘价'] > df[level]).astype(int)

        # 成本集中度
        df['cost_range_5_95'] = df['95分位成本'] - df['5分位成本']
        df['cost_concentration'] = df['cost_range_5_95'] / df['加权平均成本']

        # 当前价格在成本分布中的百分位（近似）
        df['cost_percentile'] = 0
        conditions = [
            df['收盘价'] <= df['5分位成本'],
            (df['收盘价'] > df['5分位成本']) & (df['收盘价'] <= df['15分位成本']),
            (df['收盘价'] > df['15分位成本']) & (df['收盘价'] <= df['50分位成本']),
            (df['收盘价'] > df['50分位成本']) & (df['收盘价'] <= df['85分位成本']),
            (df['收盘价'] > df['85分位成本']) & (df['收盘价'] <= df['95分位成本']),
            df['收盘价'] > df['95分位成本']
        ]
        choices = [0.025, 0.10, 0.325, 0.675, 0.90, 0.975]
        df['cost_percentile'] = np.select(conditions, choices, default=0.5)

        return df

    def calculate_market_sentiment(self, df):
        """计算市场情绪特征"""
        # 胜率衍生特征
        df['win_rate_momentum'] = df['胜率'].pct_change(5)
        df['win_rate_ma'] = df['胜率'].rolling(10).mean()

        # 价格-胜率背离
        df['price_win_correlation'] = df['收盘价'].rolling(10).corr(df['胜率'])

        # 支撑压力强度
        df['support_strength'] = (df['收盘价'] - df['5分位成本']) / (df['95分位成本'] - df['5分位成本'] + 1e-8)
        df['resistance_strength'] = (df['95分位成本'] - df['收盘价']) / (df['95分位成本'] - df['5分位成本'] + 1e-8)

        return df

    def prepare_features(self, df):
        """准备所有特征"""
        df = self.calculate_technical_indicators(df.copy())
        df = self.calculate_cost_pressure_features(df.copy())
        df = self.calculate_market_sentiment(df.copy())

        # 选择特征列
        feature_columns = [
            'price_change', 'momentum_5', 'momentum_10', 'volatility_5', 'volatility_10',
            'high_low_ratio', 'close_position', 'volume_ratio',
            'price_to_5分位成本', 'price_to_15分位成本', 'price_to_50分位成本',
            'price_to_85分位成本', 'price_to_95分位成本', 'above_5分位成本',
            'above_15分位成本', 'above_50分位成本', 'above_85分位成本', 'above_95分位成本',
            'cost_concentration', 'cost_percentile', 'win_rate_momentum',
            'win_rate_ma', 'price_win_correlation', 'support_strength', 'resistance_strength'
        ]

        # 目标变量：下一日的收盘价
        df['target'] = df['收盘价'].shift(-1)

        return df[feature_columns], df['target']

    def initial_train(self, df):
        """初始训练（前180天）"""
        print("进行初始模型训练...")

        # 准备特征和目标
        features, target = self.prepare_features(df.iloc[:self.lookback_days].copy())
        self.feature_names = features.columns.tolist()

        # 移除包含NaN的行
        valid_idx = features.notna().all(axis=1) & target.notna()
        features = features[valid_idx]
        target = target[valid_idx]

        if len(features) < 50:
            raise ValueError("有效训练数据不足，无法进行可靠训练")

        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)

        # 训练随机森林模型
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.model.fit(features_scaled, target)

        # 初始预测评估
        train_pred = self.model.predict(features_scaled)
        train_rmse = np.sqrt(mean_squared_error(target, train_pred))
        train_mae = mean_absolute_error(target, train_pred)

        print(f"初始训练完成 - 训练集RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")

        return True

    def predict_next_day(self, current_data):
        """预测下一天"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用initial_train方法")

        # 准备特征
        features, _ = self.prepare_features(current_data)
        latest_features = features.iloc[[-1]].copy()

        # 确保特征完整
        if latest_features.isna().any().any():
            print("警告：最新数据包含缺失值，使用前值填充")
            latest_features = latest_features.ffill()

        # 标准化并预测
        features_scaled = self.scaler.transform(latest_features)
        prediction = self.model.predict(features_scaled)[0]

        return prediction

    def online_learning_update(self, new_actual_price, current_data):
        """在线学习更新"""
        self.actual_history.append(new_actual_price)

        if len(self.actual_history) > 5:  # 积累一定新数据后更新
            # 准备更新数据
            update_start = max(0, len(current_data) - 10)  # 最近10天数据
            update_data = current_data.iloc[update_start:]

            features, target = self.prepare_features(update_data)
            valid_idx = features.notna().all(axis=1) & target.notna()
            features = features[valid_idx]
            target = target[valid_idx]

            if len(features) > 3:
                features_scaled = self.scaler.transform(features)

                # 部分拟合更新（在实际应用中可能需要使用warm_start）
                # 这里简化处理：用新数据重新训练
                try:
                    self.model.fit(features_scaled, target)
                    print(f"模型已更新，使用{len(features)}个新样本")
                except:
                    print("模型更新失败，继续使用原模型")


def run_prediction_simulation(full_data):
    """
    运行完整的预测模拟
    """
    print("开始证券价格预测模拟...")
    print(f"数据总量: {len(full_data)}天")
    print(f"训练期: 180天, 预测期: {len(full_data) - 180}天")

    # 初始化预测器
    predictor = AdaptiveCPTPredictor(lookback_days=180)

    # 初始训练
    success = predictor.initial_train(full_data)
    if not success:
        print("初始训练失败")
        return

    # 逐日预测
    predictions = []
    actuals = []
    dates = []

    for day in range(180, len(full_data)):
        current_date = full_data.index[day]
        current_data_snapshot = full_data.iloc[:day]  # 到当前日期的所有数据

        try:
            # 预测下一天
            pred_price = predictor.predict_next_day(current_data_snapshot)
            actual_price = full_data.iloc[day]['收盘价']

            predictions.append(pred_price)
            actuals.append(actual_price)
            dates.append(current_date)

            # 输出预测结果
            error = pred_price - actual_price
            error_pct = error / actual_price * 100

            print(f"日期: {current_date.strftime('%Y-%m-%d')}")
            print(f"  预测: {pred_price:.4f}, 实际: {actual_price:.4f}, 误差: {error_pct:+.2f}%")

            # 在线学习更新（每5天或误差较大时）
            if len(predictions) % 5 == 0 or abs(error_pct) > 2.0:
                predictor.online_learning_update(actual_price, current_data_snapshot)

        except Exception as e:
            print(f"日期 {current_date} 预测失败: {e}")
            continue

    # 评估最终性能
    if len(predictions) > 0:
        final_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        final_mae = mean_absolute_error(actuals, predictions)
        accuracy = np.mean(np.sign(np.array(predictions[1:]) - np.array(predictions[:-1])) ==
                           np.sign(np.array(actuals[1:]) - np.array(actuals[:-1]))) * 100

        print("\n" + "=" * 50)
        print("预测模拟完成")
        print(f"最终性能评估:")
        print(f"RMSE: {final_rmse:.4f}")
        print(f"MAE: {final_mae:.4f}")
        print(f"方向准确率: {accuracy:.2f}%")

        return {
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'metrics': {
                'rmse': final_rmse,
                'mae': final_mae,
                'direction_accuracy': accuracy
            }
        }


# 示例使用
def example_usage():
    """
    示例：如何使用这个预测系统
    """
    # 假设您有一个DataFrame，包含200天的数据，列名如下：
    # ['开盘价', '收盘价', '最高价', '最低价', '成交量',
    #  '5分位成本', '15分位成本', '50分位成本', '85分位成本', '95分位成本',
    #  '加权平均成本', '胜率']

    # 生成模拟数据用于演示
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # 模拟价格数据（随机游走）
    prices = [100]
    for i in range(1, 200):
        change = np.random.normal(0, 0.02)  # 日均2%波动
        prices.append(prices[-1] * (1 + change))

    mock_data = pd.DataFrame({
        '开盘价': prices,
        '收盘价': prices,
        '最高价': [p * 1.02 for p in prices],
        '最低价': [p * 0.98 for p in prices],
        '成交量': np.random.lognormal(10, 1, 200),
        '5分位成本': [p * 0.85 for p in prices],
        '15分位成本': [p * 0.90 for p in prices],
        '50分位成本': [p * 0.95 for p in prices],
        '85分位成本': [p * 1.05 for p in prices],
        '95分位成本': [p * 1.10 for p in prices],
        '加权平均成本': [p * 0.98 for p in prices],
        '胜率': np.random.uniform(0.3, 0.7, 200)
    }, index=dates)

    # 运行预测模拟
    results = run_prediction_simulation(mock_data)

    return results

# 运行示例
# results = example_usage()