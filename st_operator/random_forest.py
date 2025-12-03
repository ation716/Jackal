"""
随机森林模型
初步实验，效果不佳
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from security_data import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')




class EnhancedAdaptivePredictor:
    """

    """

    def __init__(self, lookback_days=180):
        self.lookback_days = lookback_days
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.prediction_history = []
        self.actual_history = []

    def calculate_enhanced_indicators(self, df):
        """计算增强的技术指标"""
        # 原有的技术指标
        # print(df['close'])
        df['price_change'] = df[SecurityFileds.CLOSE.value].pct_change()
        df['momentum_5'] = df[SecurityFileds.CLOSE.value].pct_change(5)
        df['momentum_10'] = df[SecurityFileds.CLOSE.value].pct_change(10)
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_10'] = df['price_change'].rolling(10).std()
        df['high_low_ratio'] = (df[SecurityFileds.HIGH.value] - df[SecurityFileds.LOW.value]) / df[SecurityFileds.CLOSE.value]
        df['close_position'] = (df[SecurityFileds.CLOSE.value] - df[SecurityFileds.LOW.value]) / (df[SecurityFileds.HIGH.value] - df[SecurityFileds.LOW.value] + 1e-8)
        df['volume_ma_5'] = df[SecurityFileds.VOLUME.value].rolling(5).mean()
        df['volume_ratio'] = df[SecurityFileds.VOLUME.value] / df['volume_ma_5']

        # 🔥 新增关键指标 🔥

        # 1. 历史价格位置特征
        df['historical_high'] = df[SecurityFileds.HIGH.value].expanding().max()  
        df['historical_low'] = df[SecurityFileds.LOW.value].expanding().min()  

        # 相对于历史极值的位置
        df['to_historical_high'] = df[SecurityFileds.CLOSE.value] / df['historical_high']  # 距历史高点的距离
        df['to_historical_low'] = df[SecurityFileds.CLOSE.value] / df['historical_low']  # 距历史低点的距离
        df['in_historical_range'] = (df[SecurityFileds.CLOSE.value] - df['historical_low']) / (
                    df['historical_high'] - df['historical_low'] + 1e-8)

        # 突破信号
        df['near_historical_high'] = (df['to_historical_high'] > 0.95).astype(int)  # 接近历史高点
        df['near_historical_low'] = (df['to_historical_low'] < 1.05).astype(int)  # 接近历史低点

        # 2. 换手率特征
        # 假设总股本为1亿股（您需要替换为真实值）
        # total_shares = 100000000
        # df['turnover_rate'] = df[SecurityFileds.VOLUME.value] / total_shares * 100  # 当日换手率(%)

        # 换手率动量
        df['turnover_ma_5'] = df[SecurityFileds.TURNOVER_RATE.value ].rolling(5).mean()
        df['turnover_ratio'] = df[SecurityFileds.TURNOVER_RATE.value ] / df['turnover_ma_5']  # 换手率相对强度
        df['turnover_momentum'] = df[SecurityFileds.TURNOVER_RATE.value ].pct_change(5)  # 换手率动量

        # 3. 价格-成交量-换手率协同特征
        df['price_turnover_corr'] = df[SecurityFileds.CLOSE.value].rolling(10).corr(df[SecurityFileds.TURNOVER_RATE.value ])  # 价量相关性
        df['high_volume_breakout'] = ((df[SecurityFileds.TURNOVER_RATE.value ] > df['turnover_ma_5'] * 1.5) &
                                      (df['to_historical_high'] > 0.9)).astype(int)  # 高换手突破

        # 价格在关键位置时的换手率特征
        df['turnover_at_resistance'] = df[SecurityFileds.TURNOVER_RATE.value ] * df['near_historical_high']  # 阻力位换手
        df['turnover_at_support'] = df[SecurityFileds.TURNOVER_RATE.value ] * df['near_historical_low']  # 支撑位换手

        return df

    def calculate_cost_pressure_features(self, df):
        """计算成本压力特征（原有）"""
        cost_levels = [SecurityFileds.COST_5TH_PERCENTILE.value, SecurityFileds.COST_15TH_PERCENTILE.value, SecurityFileds.COST_50TH_PERCENTILE.value, SecurityFileds.COST_85TH_PERCENTILE.value, SecurityFileds.COST_95TH_PERCENTILE.value]

        for level in cost_levels:
            df[f'price_to_{level}'] = df[SecurityFileds.CLOSE.value] / df[level]
            df[f'above_{level}'] = (df[SecurityFileds.CLOSE.value] > df[level]).astype(int)

        df['cost_range_5_95'] = df[SecurityFileds.COST_95TH_PERCENTILE.value] - df[SecurityFileds.COST_5TH_PERCENTILE.value]
        df['cost_concentration'] = df['cost_range_5_95'] / df[SecurityFileds.WEIGHTED_AVERAGE_COST.value]

        # 成本分布百分位
        conditions = [
            df[SecurityFileds.CLOSE.value] <= df[SecurityFileds.COST_5TH_PERCENTILE.value],
            (df[SecurityFileds.CLOSE.value] > df[SecurityFileds.COST_5TH_PERCENTILE.value]) & (df[SecurityFileds.CLOSE.value] <= df[SecurityFileds.COST_15TH_PERCENTILE.value]),
            (df[SecurityFileds.CLOSE.value] > df[SecurityFileds.COST_15TH_PERCENTILE.value]) & (df[SecurityFileds.CLOSE.value] <= df[SecurityFileds.COST_50TH_PERCENTILE.value]),
            (df[SecurityFileds.CLOSE.value] > df[SecurityFileds.COST_50TH_PERCENTILE.value]) & (df[SecurityFileds.CLOSE.value] <= df[SecurityFileds.COST_85TH_PERCENTILE.value]),
            (df[SecurityFileds.CLOSE.value] > df[SecurityFileds.COST_85TH_PERCENTILE.value]) & (df[SecurityFileds.CLOSE.value] <= df[SecurityFileds.COST_95TH_PERCENTILE.value]),
            df[SecurityFileds.CLOSE.value] > df[SecurityFileds.COST_95TH_PERCENTILE.value]
        ]
        choices = [0.025, 0.10, 0.325, 0.675, 0.90, 0.975]
        df['cost_percentile'] = np.select(conditions, choices, default=0.5)

        return df

    def calculate_market_sentiment(self, df):
        """计算市场情绪特征（原有）"""
        df['win_rate_momentum'] = df[SecurityFileds.WIN_RATE.value].pct_change(5)
        df['win_rate_ma'] = df[SecurityFileds.WIN_RATE.value].rolling(10).mean()
        df['price_win_correlation'] = df[SecurityFileds.CLOSE.value].rolling(10).corr(df[SecurityFileds.WIN_RATE.value])
        df['support_strength'] = (df[SecurityFileds.CLOSE.value] - df[SecurityFileds.COST_5TH_PERCENTILE.value]) / (df[SecurityFileds.COST_95TH_PERCENTILE.value] - df[SecurityFileds.COST_5TH_PERCENTILE.value] + 1e-8)
        df['resistance_strength'] = (df[SecurityFileds.COST_95TH_PERCENTILE.value] - df[SecurityFileds.CLOSE.value]) / (df[SecurityFileds.COST_95TH_PERCENTILE.value] - df[SecurityFileds.COST_5TH_PERCENTILE.value] + 1e-8)

        return df

    def prepare_features(self, df):
        """准备所有特征"""
        df = self.calculate_enhanced_indicators(df.copy())
        df = self.calculate_cost_pressure_features(df.copy())
        df = self.calculate_market_sentiment(df.copy())

        # 选择特征列（包含新增特征）
        feature_columns = [
            # 原有基础特征
            'price_change', 'momentum_5', 'momentum_10', 'volatility_5', 'volatility_10',
            'high_low_ratio', 'close_position', 'volume_ratio',

            # 🔥 新增历史价格特征
            'to_historical_high', 'to_historical_low', 'in_historical_range',
            'near_historical_high', 'near_historical_low',

            # 🔥 新增换手率特征
            SecurityFileds.TURNOVER_RATE.value , 'turnover_ratio', 'turnover_momentum',
            'price_turnover_corr', 'high_volume_breakout',
            'turnover_at_resistance', 'turnover_at_support',

            # 原有成本特征
            SecurityFileds.COST_5TH_PERCENTILE.value, SecurityFileds.COST_15TH_PERCENTILE.value, SecurityFileds.COST_50TH_PERCENTILE.value,
            SecurityFileds.COST_85TH_PERCENTILE.value, SecurityFileds.COST_95TH_PERCENTILE.value, f'above_{SecurityFileds.COST_5TH_PERCENTILE.value}',
            f'above_{SecurityFileds.COST_15TH_PERCENTILE.value}', f'above_{SecurityFileds.COST_50TH_PERCENTILE.value}', f'above_{SecurityFileds.COST_85TH_PERCENTILE.value}', f'above_{SecurityFileds.COST_95TH_PERCENTILE.value}',
            'cost_concentration', 'cost_percentile',

            # 原有情绪特征
            'win_rate_momentum', 'win_rate_ma', 'price_win_correlation',
            'support_strength', 'resistance_strength'
        ]

        # 目标变量：下一日的收盘价
        df['target'] = df[SecurityFileds.CLOSE.value].shift(-1)

        return df[feature_columns], df['target']

    def initial_train(self, df):
        """初始训练"""
        print("进行增强版模型训练...")

        features, target = self.prepare_features(df.iloc[:self.lookback_days].copy())
        self.feature_names = features.columns.tolist()

        # 移除包含NaN的行
        valid_idx = features.notna().all(axis=1) & target.notna()
        features = features[valid_idx]
        target = target[valid_idx]

        if len(features) < 50:
            raise ValueError("有效训练数据不足")

        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)

        # 训练增强的随机森林模型
        self.model = RandomForestRegressor(
            n_estimators=150,  # 增加树的数量
            max_depth=12,  # 增加深度以捕捉更复杂关系
            min_samples_split=3,  # 减少分裂要求
            max_features='sqrt',  # 特征采样策略
            random_state=42
        )
        self.model.fit(features_scaled, target)

        # 特征重要性分析
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("Top 10 重要特征:")
        print(feature_importance.head(10))

        # 初始预测评估
        train_pred = self.model.predict(features_scaled)
        train_rmse = np.sqrt(mean_squared_error(target, train_pred))
        train_mae = mean_absolute_error(target, train_pred)

        print(f"增强训练完成 - 训练集RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")

        return True

    # 其余方法（predict_next_day, online_learning_update等）保持不变
    # ...

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

def plot_predictions_simple(df, predictions, actuals, dates, train_days=180):
    """
    简洁版预测结果可视化
    """
    plt.figure(figsize=(14, 8))

    # 绘制完整历史价格
    plt.plot(df.index, df[SecurityFileds.CLOSE.value], label='his_price', color='blue', alpha=0.6, linewidth=1)

    # 绘制预测期价格
    plt.plot(dates, actuals, label='act_price', color='green', linewidth=2, marker='s', markersize=4)
    plt.plot(dates, predictions, label='pred_price', color='red', linewidth=2, marker='o', markersize=4)

    # 标记预测开始点
    prediction_start_date = df.index[train_days]
    plt.axvline(x=prediction_start_date, color='black', linestyle='--', alpha=0.8, label='prediction start')

    # 填充区域
    plt.axvspan(df.index[0], prediction_start_date, alpha=0.1, color='gray', label='training period')
    plt.axvspan(prediction_start_date, df.index[-1], alpha=0.1, color='yellow', label='prediction period')

    plt.title('prediction result', fontsize=14, fontweight='bold')
    plt.ylabel('price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 设置日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    plt.tight_layout()
    plt.show()
    time.sleep(2)

def run_prediction_simulation(full_data):
    """
    运行完整的预测模拟
    """
    print("开始证券价格预测模拟...")
    print(f"数据总量: {len(full_data)}天")
    print(f"训练期: 180天, 预测期: {len(full_data) - 180}天")

    # 初始化预测器
    predictor = EnhancedAdaptivePredictor(lookback_days=180)

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
        current_data_snapshot = full_data.iloc[:day]  # 第 0 天到第 day-1 天

        try:
            # 预测下一天
            pred_price = predictor.predict_next_day(current_data_snapshot)
            actual_price = full_data.iloc[day][SecurityFileds.CLOSE.value]

            predictions.append(pred_price)
            actuals.append(actual_price)
            dates.append(current_date)

            # 输出预测结果
            error = pred_price - actual_price
            error_pct = error / actual_price * 100

            print(f"日期: {full_data.iloc[current_date]['trade_date'].strftime('%Y%m%d')}")
            print(f"  预测: {pred_price:.4f}, 实际: {actual_price:.4f}, 误差: {error_pct:+.2f}%")

            # 在线学习更新（每5天或误差较大时）
            if len(predictions) % 5 == 0 or abs(error_pct) > 2.0:
                predictor.online_learning_update(actual_price, current_data_snapshot)

        except Exception as e:
            print(f"日期 {current_date} 预测失败: {e}")
            continue
    # 再预测一天
    current_data_snapshot = full_data.iloc[:len(full_data)-1]
    pred_price = predictor.predict_next_day(current_data_snapshot)
    print(f"下一日预测: {pred_price:.4f}")
    # 评估最终性能
    if len(predictions) > 0:
        change_pct_actual = (np.array(actuals[1:]) - np.array(actuals[:-1])) / np.array(actuals[:-1]) * 100
        change_pct_pred = (np.array(predictions[1:]) - np.array(predictions[:-1])) / np.array(predictions[:-1]) * 100
        final_rmse = np.sqrt(mean_squared_error(change_pct_actual, change_pct_pred))
        final_mae = mean_absolute_error(change_pct_actual, change_pct_pred)
        accuracy = np.mean(np.sign(np.array(predictions[1:]) - np.array(predictions[:-1])) ==
                           np.sign(np.array(actuals[1:]) - np.array(actuals[:-1]))) * 100

        print("\n" + "=" * 50)
        print("预测模拟完成")
        print(f"最终性能评估:")
        print(f"RMSE: {final_rmse:.4f}")
        print(f"MAE: {final_mae:.4f}")
        print(f"方向准确率: {accuracy:.2f}%")


        with open('random_forest.txt', 'w') as f:
            f.write("dates\tactual\tprediction\n")
            for i in range(len(actuals)):
                if i>0:
                    f.write("{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}\n".format(dates[i], actuals[i], predictions[i], change_pct_actual[i-1], change_pct_pred[i-1]))
                else:
                    f.write("{:<10.5f}{:<10.5f}{:<10.5f}\n".format(dates[i], actuals[i], predictions[i]))
        plot_predictions_simple(df, predictions, actuals, dates, train_days=180)
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





    # 使用示例
    # results = run_prediction_simulation(df)
    # plot_predictions_simple(df, results['predictions'], results['actuals'], results['dates'])


# # 示例使用
# def example_usage():
#     """
#     示例：如何使用这个预测系统
#     """
#     # 假设您有一个DataFrame，包含200天的数据，列名如下：
#     # ['开盘价', '收盘价', SecurityFileds.HIGH.value, SecurityFileds.LOW.value, SecurityFileds.VOLUME.value,
#     #  SecurityFileds.COST_5TH_PERCENTILE, SecurityFileds.COST_15TH_PERCENTILE, SecurityFileds.COST_50TH_PERCENTILE, SecurityFileds.COST_85TH_PERCENTILE, SecurityFileds.COST_95TH_PERCENTILE,
#     #  '加权平均成本', '胜率']
#
#     # 生成模拟数据用于演示
#     np.random.seed(42)
#     dates = pd.date_range('2023-01-01', periods=200, freq='D')
#
#     # 模拟价格数据（随机游走）
#     prices = [100]
#     for i in range(1, 200):
#         change = np.random.normal(0, 0.02)  # 日均2%波动
#         prices.append(prices[-1] * (1 + change))
#
#     mock_data = pd.DataFrame({
#         '开盘价': prices,
#         '收盘价': prices,
#         SecurityFileds.HIGH.value: [p * 1.02 for p in prices],
#         SecurityFileds.LOW.value: [p * 0.98 for p in prices],
#         SecurityFileds.VOLUME.value: np.random.lognormal(10, 1, 200),
#         SecurityFileds.COST_5TH_PERCENTILE: [p * 0.85 for p in prices],
#         SecurityFileds.COST_15TH_PERCENTILE: [p * 0.90 for p in prices],
#         SecurityFileds.COST_50TH_PERCENTILE: [p * 0.95 for p in prices],
#         SecurityFileds.COST_85TH_PERCENTILE: [p * 1.05 for p in prices],
#         SecurityFileds.COST_95TH_PERCENTILE: [p * 1.10 for p in prices],
#         '加权平均成本': [p * 0.98 for p in prices],
#         SecurityFileds.WIN_RATE: np.random.uniform(0.3, 0.7, 200)
#     }, index=dates)
#
#     # 运行预测模拟
#     results = run_prediction_simulation(mock_data)
#
#     return results

# 运行示例
# results = example_usage()

if __name__ == '__main__':
    code_list={
        # 'hmqc':'000572',
        # 'flm':'603686',
        # 'cjxc':'002171',
        # 'hfzg':'602122',
        # 'ycm':'002101',
        # 'lszz':'603169',
        # 'hjhs':'603616',
        # 'xggf':'600815',
        # 'hfzg':'603122',
        # 'smgf':'600810',
        # 'sdjt':'600734',
        # 'tytx':'002792',
        # 'lkfw':'002413'
        # 'hxsp':'002702'
        # 'dmgx':'002632',
        'dbjt':'600693'

    }
    # ts_code = "603122.SH"
    start_date = "20240924"
    end_date = "20251209"
    analyzer = ChipDistributionAnalyzer()
    counter=0
    for name,ts_code in code_list.items():

        df1 = analyzer.get_daily_ak(ts_code, start_date, end_date)
        df1 = df1.rename(columns={'换手率': 'turnover_rate'})
        df2 = analyzer.get_stock_chip_distribution(ts_code, start_date, end_date)
        df3 = analyzer.get_daily_tu(ts_code, start_date, end_date)
        df2 = df2.iloc[::-1].reset_index(drop=True)
        df3 = df3.iloc[::-1].reset_index(drop=True)
        combined_df = pd.concat([df2, df3.iloc[:, 2:11], df1.iloc[:, 11:12]], axis=1)
        combined_df.to_csv(f'../results/stacks/{name}.csv',
                  index=False,  # 不保存索引
                  encoding='utf_8_sig',  # 支持中文
                  sep=',')  # 分隔符
        counter+=1
        if counter % 5 == 0:
            time.sleep(60)
    # df = pd.read_csv('data_demo.csv',
    #                  encoding='utf_8_sig',  # 中文编码
    #                  parse_dates=['trade_date'])  # 解析日期列
    # # df['trade_date']=pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    # results = run_prediction_simulation(df)