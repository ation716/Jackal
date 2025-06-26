import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Multiply, LayerNormalization
import numpy as np
from q_save import DataStorage
import h5py

def create_model(input_shape, num_stocks):
    inputs = Input(shape=input_shape)

    # 板块关联性编码层
    sector_layer = LSTM(64, return_sequences=True)(inputs)
    sector_layer = LayerNormalization()(sector_layer)

    # 个股特征提取
    stock_layers = []
    for _ in range(num_stocks):
        lstm_out = LSTM(32, return_sequences=True)(inputs)
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = tf.nn.softmax(attention, axis=1)
        context = Multiply()([lstm_out, attention])
        stock_layers.append(context)

    # 合并板块和个股信息
    merged = tf.concat([sector_layer] + stock_layers, axis=-1)

    # 事件冲击模拟层
    event_impact = Dense(16, activation='relu')(merged)
    decay_factor = tf.keras.layers.Lambda(
        lambda x: tf.exp(-0.1 * tf.range(tf.shape(x)[1], dtype=tf.float32)))
    event_impact = Multiply()([event_impact, decay_factor])

    # 输出层
    output = Dense(num_stocks)(event_impact[:, -1, :])

    return Model(inputs=inputs, outputs=output)


class AdaptiveWeightAdjuster:
    def __init__(self, initial_weights):
        self.weights = initial_weights
        self.error_history = []
        self.window_size = 10

    def update_weights(self, current_errors):
        self.error_history.append(current_errors)
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)

        # 计算误差趋势
        error_trend = np.mean(np.array(self.error_history), axis=0)

        # 调整权重 - 误差大的股票权重降低
        adjustment = 1 / (1 + error_trend)
        self.weights = self.weights * adjustment
        self.weights = self.weights / np.sum(self.weights)  # 归一化

        return self.weights


class StockPredictor:
    def __init__(self, stock_list):
        self.storage = DataStorage()
        self.model = create_model(input_shape=(30, len(stock_list)),
                                  num_stocks=len(stock_list))
        self.adjuster = AdaptiveWeightAdjuster(
            initial_weights=np.ones(len(stock_list)) / len(stock_list))

    def fetch_and_store(self):
        # 使用akshare获取数据
        new_data = {}  # 实际替换为akshare获取的数据
        self.storage.append_data(new_data)

    def prepare_training_data(self):
        # 从HDF5文件加载数据并预处理
        with h5py.File(self.storage.filename, 'r') as f:
            # 实现数据加载和窗口创建逻辑
            pass
        return X_train, y_train

    def train_model(self):
        X, y = self.prepare_training_data()
        # 实现训练逻辑，包括时间衰减加权
        # 最近数据权重更高
        time_weights = np.exp(np.linspace(0, 1, X.shape[1]))

    def predict(self):
        # 获取最新30个时间点数据
        latest_data = self.get_latest_window()
        predictions = self.model.predict(latest_data)

        # 添加随机波动性
        volatility = np.std(latest_data, axis=1)
        predictions += np.random.normal(0, volatility * 0.2, predictions.shape)

        return predictions

    def adaptive_learning(self, real_values):
        predictions = self.predict()
        errors = np.abs(predictions - real_values)
        new_weights = self.adjuster.update_weights(errors)

        # 使用新权重重新训练
        self.train_model(weights=new_weights)