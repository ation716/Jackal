import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import akshare as ak
import warnings
warnings.filterwarnings('ignore')

# 假设已经获取了三年数据，数据结构如下

data = {
    'code': ['600000', '600001', ...],  # 股票代码
    'name': ['浦发银行', '邯郸钢铁', ...],  # 股票名称
    'date': ['2020-01-02', '2020-01-03', ...],  # 交易日
    'close': [10.2, 15.3, ...],  # 收盘价
    'pe': [12.5, 8.7, ...],  # 市盈率
    'pb': [1.2, 0.8, ...],  # 市净率
    'rsi_6': [45, 60, ...],  # 6日RSI
    'rsi_12': [50, 55, ...],  # 12日RSI
    'turnover': [3.5, 2.8, ...],  # 换手率(%)
    'market_cap': [500, 300, ...],  # 市值(亿)
    'revenue_growth': [0.15, -0.03, ...],  # 营收增长率
    'profit_growth': [0.2, 0.1, ...],  # 利润增长率
    'is_doubled': [0, 1, ...]  # 是否翻倍(未来3个月)
}

# 实际使用时替换为从akshare获取的真实数据
df = pd.DataFrame(data)

# 标记翻倍股
df['doubled_date'] = np.where(df['is_doubled']==1, df['date'], np.nan)
df['doubled_date'] = pd.to_datetime(df['doubled_date'])


def analyze_periodicity(df):
    # 提取翻倍月份
    doubled_stocks = df[df['is_doubled'] == 1].copy()
    doubled_stocks['month'] = doubled_stocks['doubled_date'].dt.month

    # 统计每月翻倍股数量
    monthly_counts = doubled_stocks['month'].value_counts().sort_index()

    # 绘制分布图
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=monthly_counts.index, y=monthly_counts.values, color='steelblue')

    # 拟合正态分布
    mu, std = norm.fit(doubled_stocks['month'])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std) * len(doubled_stocks) * 1.5  # 缩放因子

    # 绘制正态曲线
    ax2 = ax.twinx()
    ax2.plot(x, p, 'r--', linewidth=2)
    ax2.set_ylabel('Probability Density', color='r')

    plt.title('翻倍股月份分布及正态拟合曲线')
    plt.xlabel('Month')
    ax.set_ylabel('Number of Doubled Stocks')
    plt.grid(True)
    plt.show()

    # 输出周期性结论
    peak_month = int(round(mu))
    print(f"翻倍股最可能出现的月份: {peak_month}月 (μ={mu:.1f}, σ={std:.1f})")

    return monthly_counts, (mu, std)


monthly_dist, params = analyze_periodicity(df)


def analyze_features(df):
    # 分离翻倍股和非翻倍股
    doubled = df[df['is_doubled'] == 1]
    non_doubled = df[df['is_doubled'] == 0]

    # 特征列表
    features = ['pe', 'pb', 'rsi_6', 'rsi_12', 'turnover',
                'market_cap', 'revenue_growth', 'profit_growth']

    # 绘制特征分布对比
    plt.figure(figsize=(16, 12))
    for i, feat in enumerate(features):
        plt.subplot(3, 3, i + 1)
        sns.kdeplot(doubled[feat], label='Doubled', shade=True)
        sns.kdeplot(non_doubled[feat], label='Non-doubled', shade=True)
        plt.title(f'{feat} Distribution')
        plt.legend()
    plt.tight_layout()
    plt.show()

    # 计算特征统计量
    stats = pd.DataFrame()
    stats['doubled_mean'] = doubled[features].mean()
    stats['non_doubled_mean'] = non_doubled[features].mean()
    stats['difference'] = stats['doubled_mean'] - stats['non_doubled_mean']
    stats['percent_diff'] = stats['difference'] / stats['non_doubled_mean'] * 100

    # 输出显著特征
    significant_features = stats[abs(stats['percent_diff']) > 20].index.tolist()
    print("\n显著差异特征:", significant_features)

    return stats


feature_stats = analyze_features(df)


def build_prediction_model(df):
    # 特征工程
    features = ['pe', 'pb', 'rsi_6', 'rsi_12', 'turnover',
                'market_cap', 'revenue_growth', 'profit_growth']

    # 使用过去5天的特征平均值
    recent_data = df.groupby('code').apply(lambda x: x.iloc[-5:][features].mean())
    recent_data['is_doubled'] = df.groupby('code')['is_doubled'].max()

    # 数据平衡处理 (SMOTE过采样)
    from imblearn.over_sampling import SMOTE
    X = recent_data[features]
    y = recent_data['is_doubled']
    X_res, y_res = SMOTE().fit_resample(X, y)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # 训练模型 (使用校准分类器以获得可靠概率)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    calibrated_clf = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
    calibrated_clf.fit(X_train, y_train)

    # 评估模型
    y_pred = calibrated_clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return calibrated_clf, scaler


model, scaler = build_prediction_model(df)


def predict_recent_doubled_stocks(model, scaler, recent_week_data):
    # 预处理最近一周数据
    features = ['pe', 'pb', 'rsi_6', 'rsi_12', 'turnover',
                'market_cap', 'revenue_growth', 'profit_growth']

    # 计算各指标5日均值
    recent_avg = recent_week_data.groupby('code')[features].mean()

    # 标准化
    X_recent = scaler.transform(recent_avg)

    # 预测概率
    probs = model.predict_proba(X_recent)[:, 1]
    recent_avg['doubled_prob'] = probs

    # 筛选概率>30%的股票
    high_prob_stocks = recent_avg[recent_avg['doubled_prob'] > 0.3]
    high_prob_stocks = high_prob_stocks.sort_values('doubled_prob', ascending=False)

    # 关联股票名称
    stock_names = recent_week_data[['code', 'name']].drop_duplicates().set_index('code')
    results = high_prob_stocks.join(stock_names, how='left')

    return results[['name', 'doubled_prob'] + features]


# 假设 recent_week_data 是最近5个交易日的数据
# 实际使用时替换为从akshare获取的最新数据
recent_data = df.groupby('code').apply(lambda x: x.iloc[-5:]).reset_index(drop=True)
predictions = predict_recent_doubled_stocks(model, scaler, recent_data)

print("\n高概率翻倍股预测结果:")
print(predictions)


def full_analysis_pipeline(historical_data, recent_week_data):
    print("=== 翻倍股周期性分析 ===")
    monthly_dist, params = analyze_periodicity(historical_data)

    print("\n=== 翻倍股特征分析 ===")
    feature_stats = analyze_features(historical_data)

    print("\n=== 构建预测模型 ===")
    model, scaler = build_prediction_model(historical_data)

    print("\n=== 近期股票预测 ===")
    predictions = predict_recent_doubled_stocks(model, scaler, recent_week_data)

    return {
        'monthly_distribution': monthly_dist,
        'normal_params': params,
        'feature_statistics': feature_stats,
        'predictions': predictions
    }


# 执行完整分析
results = full_analysis_pipeline(df, recent_data)