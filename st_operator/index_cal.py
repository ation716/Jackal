# 指标计算，主要计算如下指标
"""
短期、中期、长期移动平均线（MA5、MA10、MA20）

相对强弱指数（RSI）

MACD指标（MACD线和信号线）

布林带（上轨、下轨）

成交量均线（Volume MA5）和成交量比率（Volume Ratio）

价格在布林带中的位置（Price Position）
"""
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib  # 技术指标计算库

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class FreeTechnicalAnalyzer:
    def __init__(self):
        """初始化，完全免费无需token"""
        pass

    def get_stock_data(self, symbol, start_date, end_date, adjust="qfq"):
        """
        获取股票数据 - 使用AkShare（完全免费）
        """
        try:
            # 自动识别市场
            if symbol.startswith(('6', '9')):
                stock_symbol = f"sh{symbol}"
            else:
                stock_symbol = f"sz{symbol}"

            # 获取历史行情数据
            stock_data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",  # 日线数据
                start_date=start_date,
                end_date=end_date,
                adjust=adjust  # 前复权
            )

            if stock_data is not None and not stock_data.empty:
                # 重命名列
                stock_data = stock_data.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'pct_chg',
                    '涨跌额': 'change',
                    '换手率': 'turnover'
                })

                stock_data['date'] = pd.to_datetime(stock_data['date'])
                stock_data = stock_data.sort_values('date')
                stock_data = stock_data.reset_index(drop=True)

                print(f"成功获取 {symbol} 的 {len(stock_data)} 条数据")
                return stock_data
            else:
                print("未获取到数据")
                return None

        except Exception as e:
            print(f"获取数据失败: {e}")
            return None

    def calculate_all_indicators(self, df):
        """
        计算所有技术指标
        """
        if df is None or df.empty:
            return None

        df = df.copy()

        # 1. 移动平均线
        df = self.calculate_moving_averages(df)

        # 2. 相对强弱指数 (RSI)
        df = self.calculate_rsi(df)

        # 3. MACD指标
        df = self.calculate_macd(df)

        # 4. 布林带
        df = self.calculate_bollinger_bands(df)

        # 5. 成交量指标
        df = self.calculate_volume_indicators(df)

        # 6. 价格在布林带中的位置
        df = self.calculate_bollinger_position(df)

        # 7. 其他常用指标
        df = self.calculate_additional_indicators(df)

        return df

    def calculate_moving_averages(self, df):
        """
        计算移动平均线 - 使用talib
        """
        # 方法1：使用talib（更准确）
        try:
            df['ma5'] = talib.MA(df['close'], timeperiod=5)
            df['ma10'] = talib.MA(df['close'], timeperiod=10)
            df['ma20'] = talib.MA(df['close'], timeperiod=20)
            df['ma60'] = talib.MA(df['close'], timeperiod=60)
        except:
            # 方法2：使用pandas（备用）
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma10'] = df['close'].rolling(10).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma60'] = df['close'].rolling(60).mean()

        # 计算价格与均线的相对位置
        df['price_vs_ma5'] = (df['close'] - df['ma5']) / df['ma5'] * 100
        df['price_vs_ma20'] = (df['close'] - df['ma20']) / df['ma20'] * 100

        return df

    def calculate_rsi(self, df, period=6):
        """
        计算相对强弱指数 (RSI) - 使用talib
        """
        try:
            df['rsi'] = talib.RSI(df['close'], timeperiod=period)
        except:
            # 手动计算RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

        # 添加RSI超买超卖线
        df['rsi_overbought'] = 70
        df['rsi_oversold'] = 30

        return df

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """
        计算MACD指标 - 使用talib
        """
        try:
            macd, macd_signal, macd_hist = talib.MACD(df['close'],
                                                      fastperiod=fast,
                                                      slowperiod=slow,
                                                      signalperiod=signal)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
        except:
            # 手动计算MACD
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=signal).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """
        计算布林带 - 使用talib
        """
        try:
            upper, middle, lower = talib.BBANDS(df['close'],
                                                timeperiod=period,
                                                nbdevup=std_dev,
                                                nbdevdn=std_dev)
            df['boll_upper'] = upper
            df['boll_middle'] = middle
            df['boll_lower'] = lower
        except:
            # 手动计算布林带
            df['boll_middle'] = df['close'].rolling(period).mean()
            rolling_std = df['close'].rolling(period).std()
            df['boll_upper'] = df['boll_middle'] + (rolling_std * std_dev)
            df['boll_lower'] = df['boll_middle'] - (rolling_std * std_dev)

        df['boll_width'] = df['boll_upper'] - df['boll_lower']

        return df

    def calculate_volume_indicators(self, df):
        """
        计算成交量指标
        """
        # 成交量移动平均
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma10'] = df['volume'].rolling(10).mean()

        # 成交量比率（当日成交量/5日均量）
        df['volume_ratio'] = df['volume'] / df['volume_ma5']

        # 量比（当日成交量/过去5日平均成交量）
        df['volume_relative'] = df['volume'] / df['volume'].rolling(5).mean()

        return df

    def calculate_bollinger_position(self, df):
        """
        计算价格在布林带中的位置
        """
        if 'boll_upper' in df.columns and 'boll_lower' in df.columns:
            df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])

            # 布林带突破信号
            df['boll_break_upper'] = (df['close'] > df['boll_upper']).astype(int)
            df['boll_break_lower'] = (df['close'] < df['boll_lower']).astype(int)

        return df

    def calculate_additional_indicators(self, df):
        """
        计算其他常用技术指标
        """
        # KD指标 (Stochastic Oscillator)
        try:
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
            df['kdj_k'] = slowk
            df['kdj_d'] = slowd
            df['kdj_j'] = 3 * slowk - 2 * slowd
        except:
            pass

        # OBV能量潮
        try:
            df['obv'] = talib.OBV(df['close'], df['volume'])
        except:
            pass

        # CCI商品路径指标
        try:
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        except:
            pass

        return df

    def get_technical_analysis_report(self, symbol, start_date, end_date):
        """
        获取完整的技术分析报告
        """
        print(f"正在分析 {symbol} 的技术指标...")

        # 获取数据
        stock_data = self.get_stock_data(symbol, start_date, end_date)

        if stock_data is None:
            return None

        # 计算技术指标
        tech_data = self.calculate_all_indicators(stock_data)

        if tech_data is not None:
            # 显示最新指标
            latest = tech_data.iloc[-1]
            print(f"\n=== {symbol} 最新技术指标 ===")
            print(f"分析日期: {latest['date'].strftime('%Y-%m-%d')}")
            print(f"收盘价: {latest['close']:.2f}")

            print(f"\n📈 移动平均线:")
            print(f"  MA5: {latest.get('ma5', 'N/A'):.2f}")
            print(f"  MA10: {latest.get('ma10', 'N/A'):.2f}")
            print(f"  MA20: {latest.get('ma20', 'N/A'):.2f}")

            print(f"\n🎯 动量指标:")
            print(f"  RSI(14): {latest.get('rsi', 'N/A'):.2f}")
            print(f"  MACD: {latest.get('macd', 'N/A'):.4f}")
            print(f"  MACD信号: {latest.get('macd_signal', 'N/A'):.4f}")

            print(f"\n📊 布林带:")
            print(f"  上轨: {latest.get('boll_upper', 'N/A'):.2f}")
            print(f"  中轨: {latest.get('boll_middle', 'N/A'):.2f}")
            print(f"  下轨: {latest.get('boll_lower', 'N/A'):.2f}")
            print(f"  位置: {latest.get('boll_position', 'N/A'):.2%}")

            print(f"\n💰 成交量:")
            print(f"  成交量: {latest['volume']:,.0f}")
            print(f"  量比: {latest.get('volume_ratio', 'N/A'):.2f}")
            print(f"  换手率: {latest.get('turnover', 'N/A'):.2f}%")

            # 技术信号汇总
            self.generate_trading_signals(latest)

            return tech_data
        else:
            print("技术指标计算失败")
            return None

    def generate_trading_signals(self, latest_data):
        """
        生成交易信号
        """
        signals = []

        # RSI信号
        rsi = latest_data.get('rsi', 50)
        if rsi > 70:
            signals.append("RSI超买")
        elif rsi < 30:
            signals.append("RSI超卖")

        # MACD信号
        macd = latest_data.get('macd', 0)
        macd_signal = latest_data.get('macd_signal', 0)
        if macd > macd_signal:
            signals.append("MACD金叉")
        else:
            signals.append("MACD死叉")

        # 布林带信号
        boll_position = latest_data.get('boll_position', 0.5)
        if boll_position > 0.8:
            signals.append("布林带上轨压力")
        elif boll_position < 0.2:
            signals.append("布林带下轨支撑")

        # 均线信号
        close_price = latest_data.get('close', 0)
        ma5 = latest_data.get('ma5', close_price)
        ma20 = latest_data.get('ma20', close_price)

        if close_price > ma5 > ma20:
            signals.append("多头排列")
        elif close_price < ma5 < ma20:
            signals.append("空头排列")

        print(f"\n🚦 技术信号: {', '.join(signals) if signals else '中性'}")

    def visualize_technical_analysis(self, tech_data, symbol):
        """
        可视化技术分析结果
        """
        if tech_data is None:
            return

        # 创建子图
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'{symbol} 技术分析图表', fontsize=16)

        # 1. 价格和移动平均线
        axes[0].plot(tech_data['date'], tech_data['close'], label='收盘价', linewidth=2, color='black')
        axes[0].plot(tech_data['date'], tech_data['ma5'], label='MA5', linestyle='--', alpha=0.7)
        axes[0].plot(tech_data['date'], tech_data['ma10'], label='MA10', linestyle='--', alpha=0.7)
        axes[0].plot(tech_data['date'], tech_data['ma20'], label='MA20', linestyle='--', alpha=0.7)
        axes[0].fill_between(tech_data['date'], tech_data['boll_upper'], tech_data['boll_lower'],
                             alpha=0.2, label='布林带')
        axes[0].set_title('价格与移动平均线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. MACD
        axes[1].plot(tech_data['date'], tech_data['macd'], label='MACD', color='blue')
        axes[1].plot(tech_data['date'], tech_data['macd_signal'], label='信号线', color='red')
        axes[1].bar(tech_data['date'], tech_data['macd_hist'], label='MACD柱', alpha=0.3)
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[1].set_title('MACD指标')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. RSI
        axes[2].plot(tech_data['date'], tech_data['rsi'], label='RSI', color='purple')
        axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='超买线')
        axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='超卖线')
        axes[2].axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        axes[2].set_ylim(0, 100)
        axes[2].set_title('RSI指标')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 4. 成交量
        axes[3].bar(tech_data['date'], tech_data['volume'], alpha=0.7, label='成交量')
        axes[3].plot(tech_data['date'], tech_data['volume_ma5'], label='成交量MA5', color='orange')
        axes[3].set_title('成交量')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        print("最后数据",tech_data['ma5'].iloc[-1], tech_data['ma10'].iloc[-1], tech_data['ma20'].iloc[-1],tech_data['rsi'].iloc[-1])
        plt.show()



# 使用示例
def main():
    # 初始化分析器（完全免费）
    analyzer = FreeTechnicalAnalyzer()

    # 分析股票（示例：宁德时代）
    symbol = "002115"

    # 获取技术分析报告
    tech_data = analyzer.get_technical_analysis_report(
        symbol=symbol,
        start_date="20250501",
        end_date="20250930"
    )

    if tech_data is not None:
        # 可视化结果
        analyzer.visualize_technical_analysis(tech_data, symbol)

        # 保存数据到CSV
        tech_data.to_csv(f'{symbol}_technical_analysis.csv', index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到 {symbol}_technical_analysis.csv")

    return tech_data


if __name__ == "__main__":
    result = main()