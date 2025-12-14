import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from math import radians
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow_addons as tfa
from cond_rnn import ConditionalRecurrent
from sklearn.metrics.pairwise import haversine_distances
from itertools import combinations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# 消融实验配置
ablation_configs = {
    "CBAM_full": {
        "use_dp_condition": True,
        "use_attention": True,
        "use_weather": True,
        "use_bilstm": True,
        "dp_concatenate": False
    }
    "CBAM_no_dp_condition": {
        "use_dp_condition": False,
        "use_attention": True,
        "use_weather": True,
        "use_bilstm": True,
        "dp_concatenate": False
    },
    "CBAM_dp_concatenate": {
        "use_dp_condition": False,
        "use_attention": True,
        "use_weather": True,
        "use_bilstm": True,
        "dp_concatenate": True
    },
    "CBAM_no_attention": {
        "use_dp_condition": True,
        "use_attention": False,
        "use_weather": True,
        "use_bilstm": True,
        "dp_concatenate": False
    },
    "CBAM_no_weather": {
        "use_dp_condition": True,
        "use_attention": True,
        "use_weather": False,
        "use_bilstm": True,
        "dp_concatenate": False
    },
    "CBAM_unidirectional": {
        "use_dp_condition": True,
        "use_attention": True,
        "use_weather": True,
        "use_bilstm": False,
        "dp_concatenate": False
    },
}

# 全局参数
max_traj_len = 81
extra_dim_full = 9  # 完整特征维度
extra_dim_no_weather = 3  # 无天气特征：daytype, hour_sin, hour_cos
dp_dim = 3
level = 8
lstm_units = 256
dense_extra_units = 18
dense_concat_units = 1024


class attention2(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(attention2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'step_dim': self.step_dim,
            'W_regularizer': self.W_regularizer,
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'b_constraint': self.b_constraint,
            'bias': self.bias
        })
        return config


def load_train_data(per=None):
    train_data = pd.read_csv("../training data/" + per + "/train_" + per + "_all_level8_new_clusters_1.csv")
    val_data = pd.read_csv("../validation data/" + per + "/val_" + per + "_all_level8_new_clusters_1.csv")
    return train_data, val_data


def load_test_data(per):
    test_data = pd.read_csv("../testing data/" + per + "0%/test_" + per + "0%_all_level8_new_clusters_1.csv")
    return test_data


def build_ablation_model(config):
    """根据消融实验配置构建模型"""
    input_gps = Input(shape=(max_traj_len, 2), name='input_gps')
    input_dp = Input(shape=(dp_dim,), name='input_dp')

    # 根据配置确定特征维度
    if config["use_weather"]:
        extra_dim = extra_dim_full
        input_extra = Input(shape=(extra_dim,), name='input_extra')
    else:
        extra_dim = extra_dim_no_weather
        input_extra = Input(shape=(extra_dim,), name='input_extra')

    # 动态价格处理方式
    if config["use_dp_condition"]:
        # 条件循环机制（原始CBAM）
        forward_lstm_gps = ConditionalRecurrent(
            LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        backward_lstm_gps = ConditionalRecurrent(
            LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01),
                 go_backwards=True))
        bilstm_inputs = (input_gps, input_dp)
    else:
        # 普通LSTM
        forward_lstm_gps = LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01))
        backward_lstm_gps = LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                 go_backwards=True)
        bilstm_inputs = input_gps

    # BiLSTM或单向LSTM
    if config["use_bilstm"]:
        bilstm_gps = Bidirectional(layer=forward_lstm_gps, backward_layer=backward_lstm_gps)(inputs=bilstm_inputs)
    else:
        bilstm_gps = forward_lstm_gps(inputs=bilstm_inputs)

    # 注意力机制
    if config["use_attention"]:
        attention_output = attention2(max_traj_len)(bilstm_gps)
    else:
        # 无注意力：使用最后一个时间步
        attention_output = bilstm_gps[:, -1, :]

    # 辅助特征处理
    dense_extra = Dense(dense_extra_units, activation='relu')(input_extra)

    # 拼接层 - 处理动态价格拼接情况
    if config["dp_concatenate"]:
        # 动态价格直接拼接
        concat = Concatenate()([attention_output, dense_extra, input_dp])
    else:
        # 原始拼接方式
        concat = Concatenate()([attention_output, dense_extra])

    # 输出层
    dense_concat = Dense(dense_concat_units, activation='relu')(concat)
    dropout_concat = Dropout(0.5)(dense_concat)
    softmax_concat = Dense(965, activation='softmax')(dropout_concat)

    # 构建模型
    if config["use_dp_condition"] or config["dp_concatenate"]:
        model = Model(inputs=[input_gps, input_dp, input_extra], outputs=softmax_concat)
    else:
        model = Model(inputs=[input_gps, input_extra], outputs=softmax_concat)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5),
                           tfa.metrics.F1Score(num_classes=965, average='macro')])
    return model


def normalize(data, level):
    for i in range(len(data)):
        data[i] = [[k[0] / pow(2, level), k[1] / pow(2, level)] for k in data[i]]
    return data


def coords_padding(data, maxlen):
    padded_data = []
    for i in range(len(data)):
        traj = data[i]
        if len(traj) < maxlen:
            traj = traj + [[0, 0]] * (maxlen - len(traj))
        elif len(traj) > maxlen:
            traj = traj[:maxlen]
        traj = [[item[0], item[1]] if len(item) == 2 else [0, 0] for item in traj]
        padded_data.append(traj)
    return padded_data


def to_one_hot(data):
    return to_categorical(data, num_classes=965)


def top_k_avg_dist(logits, targets, k, center_clusters):
    values, indices = tf.math.top_k(logits, k=k, sorted=True)
    all_dist = []
    for i in range(len(indices)):
        tmp_dist = []
        for pred in indices[i]:
            pred_lon = radians(center_clusters.iloc[int(pred)]['center_lon'])
            pred_lat = radians(center_clusters.iloc[int(pred)]['center_lat'])
            true_lon = radians(center_clusters.iloc[int(targets[i])]['center_lon'])
            true_lat = radians(center_clusters.iloc[int(targets[i])]['center_lat'])
            tmp_dist.append(
                (haversine_distances([[pred_lat, pred_lon], [true_lat, true_lon]]) * 6371000 / 1000)[0][1] * 1000)
        all_dist.append(tmp_dist)
    all_dist = np.array(all_dist)
    return np.mean(all_dist), np.mean(np.median(all_dist, axis=1)), np.mean(np.std(all_dist, axis=1))


def train_single_ablation_model(config_name, config):
    """训练单个消融实验模型"""
    print(f"Training {config_name}...")

    # 创建检查点目录
    os.makedirs(f"ablation_checkpoints/{config_name}", exist_ok=True)

    # 加载数据
    train_data, val_data = load_train_data("80%")

    # 数据预处理
    train_x_gps = [eval(i) for i in train_data['coords_of_traj']]
    train_y = np.array(to_one_hot(train_data['dest_cluster']))
    val_x_gps = [eval(i) for i in val_data['coords_of_traj']]
    val_y = np.array(to_one_hot(val_data['dest_cluster']))

    train_x_gps = normalize(train_x_gps, level)
    val_x_gps = normalize(val_x_gps, level)
    train_x_gps = np.array(coords_padding(train_x_gps, max_traj_len))
    val_x_gps = np.array(coords_padding(val_x_gps, max_traj_len))

    # 特征选择
    if config["use_weather"]:
        train_x_extra = np.array(train_data[['daytype', 'weather_0', 'weather_1', 'weather_2',
                                             'weather_3', 'weather_4', 'weather_5', 'hour_sin', 'hour_cos']])
        val_x_extra = np.array(val_data[['daytype', 'weather_0', 'weather_1', 'weather_2',
                                         'weather_3', 'weather_4', 'weather_5', 'hour_sin', 'hour_cos']])
    else:
        train_x_extra = np.array(train_data[['daytype', 'hour_sin', 'hour_cos']])
        val_x_extra = np.array(val_data[['daytype', 'hour_sin', 'hour_cos']])

    train_x_dp = np.array(train_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])
    val_x_dp = np.array(val_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])

    # 构建模型
    model = build_ablation_model(config)

    # 回调函数
    checkpoint_path = f"ablation_checkpoints/{config_name}/checkpoint"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_top_k_categorical_accuracy',
        mode='max',
        save_best_only=True
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # 训练
    print(f"Starting training for {config_name}...")
    if config["use_dp_condition"] or config["dp_concatenate"]:
        history = model.fit(
            [train_x_gps, train_x_dp, train_x_extra], train_y,
            epochs=50,  # 可以先试50个epochs
            batch_size=128,
            validation_data=([val_x_gps, val_x_dp, val_x_extra], val_y),
            callbacks=[model_checkpoint, early_stopping],
            verbose=1
        )
    else:
        history = model.fit(
            [train_x_gps, train_x_extra], train_y,
            epochs=50,
            batch_size=128,
            validation_data=([val_x_gps, val_x_extra], val_y),
            callbacks=[model_checkpoint, early_stopping],
            verbose=1
        )

    print(f"✅ Finished training {config_name}")
    return history


def run_ablation_study():
    """运行所有消融实验 - 修正版本"""
    center_clusters_gps = pd.read_csv(
        "../processed_data/clusters_5_ring/mean_shift clustering/965_clusters_center_coords.csv")

    all_results = {}

    for config_name, config in ablation_configs.items():
        print(f"\n{'=' * 50}")
        print(f"Running ablation study: {config_name}")
        print(f"{'=' * 50}")

        # 构建模型
        model = build_ablation_model(config)

        # === 修改这里：加载对应配置的权重 ===
        checkpoint_path = f"ablation_checkpoints/{config_name}/checkpoint"
        try:
            model.load_weights(checkpoint_path)
            print(f"✅ Loaded trained weights for {config_name}")
        except:
            print(f"❌ No trained weights found for {config_name}")
            print(f"💡 Running training for {config_name} first...")

            # 如果没有训练权重，先训练这个配置
            train_single_ablation_model(config_name, config)

            # 重新加载权重
            try:
                model.load_weights(checkpoint_path)
                print(f"✅ Loaded newly trained weights for {config_name}")
            except:
                print(f"⚠️  Using randomly initialized model for {config_name}")
        # === 修改结束 ===

        config_results = {}

        # 测试所有前缀比例
        for per in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            print(f"Testing {per}0% prefix...")

            # 加载测试数据
            test_data = load_test_data(per)
            test_x_gps = [eval(i) for i in test_data['coords_of_traj']]
            test_y = np.array(test_data['dest_cluster'])

            # 数据预处理
            test_x_gps = normalize(test_x_gps, level)
            test_x_gps = np.array(coords_padding(test_x_gps, max_traj_len))

            # 特征选择
            if config["use_weather"]:
                test_x_extra = np.array(test_data[['daytype', 'weather_0', 'weather_1', 'weather_2',
                                                   'weather_3', 'weather_4', 'weather_5', 'hour_sin', 'hour_cos']])
            else:
                test_x_extra = np.array(test_data[['daytype', 'hour_sin', 'hour_cos']])

            test_x_dp = np.array(test_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])

            # 预测
            if config["use_dp_condition"] or config["dp_concatenate"]:
                y_test_pred = model.predict([test_x_gps, test_x_dp, test_x_extra], batch_size=128, verbose=0)
            else:
                y_test_pred = model.predict([test_x_gps, test_x_extra], batch_size=128, verbose=0)

            # 计算指标
            k_results = {}
            for k in [1, 5, 10]:
                mean, avg_median, avg_std = top_k_avg_dist(y_test_pred, test_y, k, center_clusters=center_clusters_gps)
                k_results[f'Top-{k}'] = mean
                print(f"  Top-{k} Distance mean: {mean:.2f}m")

            config_results[f'{per}0%'] = k_results

        all_results[config_name] = config_results

        # 保存每个配置的结果
        with open(f"ablation_results_{config_name}.txt", "w") as f:
            f.write(f"Ablation Study Results: {config_name}\n")
            f.write("Config: " + str(config) + "\n\n")
            for prefix, metrics in config_results.items():
                f.write(f"{prefix}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.2f}m\n")
                f.write("\n")

    # 生成对比报告
    generate_comparison_report(all_results)

    return all_results



def generate_comparison_report(all_results):
    """生成消融实验对比报告"""
    print(f"\n{'=' * 60}")
    print("ABLATION STUDY COMPARISON REPORT")
    print(f"{'=' * 60}")

    with open("ablation_comparison_report.txt", "w") as f:
        f.write("Ablation Study Comparison Report\n")
        f.write("=" * 50 + "\n\n")

        # 比较每个前缀比例下各配置的性能
        prefixes = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']

        for prefix in prefixes:
            f.write(f"\n{prefix} Prefix Results:\n")
            f.write("-" * 30 + "\n")

            prefix_key = f"{prefix[0]}0%"
            results_at_prefix = {}

            for config_name, config_results in all_results.items():
                if prefix_key in config_results:
                    top1_error = config_results[prefix_key]['Top-1']
                    results_at_prefix[config_name] = top1_error

            # 按性能排序
            sorted_results = sorted(results_at_prefix.items(), key=lambda x: x[1])

            for i, (config_name, error) in enumerate(sorted_results):
                f.write(f"{i + 1:2d}. {config_name:<25} {error:8.2f}m\n")

            # 计算相对于完整模型的性能下降
            if "CBAM_full" in results_at_prefix:
                full_model_error = results_at_prefix["CBAM_full"]
                f.write(f"\nPerformance degradation relative to CBAM_full ({full_model_error:.2f}m):\n")
                for config_name, error in results_at_prefix.items():
                    if config_name != "CBAM_full":
                        degradation = ((error - full_model_error) / full_model_error) * 100
                        f.write(f"  {config_name:<25} +{degradation:6.2f}%\n")


def train_ablation_models():
    """训练所有消融实验模型（如果需要重新训练）"""
    for config_name, config in ablation_configs.items():
        print(f"\nTraining {config_name}...")

        # 加载数据
        train_data, val_data = load_train_data("80%")

        # 数据预处理（根据配置选择特征）
        train_x_gps = [eval(i) for i in train_data['coords_of_traj']]
        train_y = np.array(to_one_hot(train_data['dest_cluster']))
        val_x_gps = [eval(i) for i in val_data['coords_of_traj']]
        val_y = np.array(to_one_hot(val_data['dest_cluster']))

        train_x_gps = normalize(train_x_gps, level)
        val_x_gps = normalize(val_x_gps, level)
        train_x_gps = np.array(coords_padding(train_x_gps, max_traj_len))
        val_x_gps = np.array(coords_padding(val_x_gps, max_traj_len))

        # 特征选择
        if config["use_weather"]:
            train_x_extra = np.array(train_data[['daytype', 'weather_0', 'weather_1', 'weather_2',
                                                 'weather_3', 'weather_4', 'weather_5', 'hour_sin', 'hour_cos']])
            val_x_extra = np.array(val_data[['daytype', 'weather_0', 'weather_1', 'weather_2',
                                             'weather_3', 'weather_4', 'weather_5', 'hour_sin', 'hour_cos']])
        else:
            train_x_extra = np.array(train_data[['daytype', 'hour_sin', 'hour_cos']])
            val_x_extra = np.array(val_data[['daytype', 'hour_sin', 'hour_cos']])

        train_x_dp = np.array(train_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])
        val_x_dp = np.array(val_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])

        # 构建和训练模型
        model = build_ablation_model(config)

        checkpoint_path = f"ablation_checkpoints/{config_name}/checkpoint"
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_top_k_categorical_accuracy',
            mode='max',
            save_best_only=True
        )

        # 训练
        if config["use_dp_condition"] or config["dp_concatenate"]:
            history = model.fit(
                [train_x_gps, train_x_dp, train_x_extra], train_y,
                epochs=200, batch_size=128,
                validation_data=([val_x_gps, val_x_dp, val_x_extra], val_y),
                callbacks=[model_checkpoint],
                verbose=1
            )
        else:
            history = model.fit(
                [train_x_gps, train_x_extra], train_y,
                epochs=200, batch_size=128,
                validation_data=([val_x_gps, val_x_extra], val_y),
                callbacks=[model_checkpoint],
                verbose=1
            )

        print(f"Finished training {config_name}")


if __name__ == "__main__":
    # 确保检查点目录存在
    os.makedirs("ablation_checkpoints", exist_ok=True)

    # 只运行非full的配置
    non_full_configs = {k: v for k, v in ablation_configs.items() if k != "CBAM_full"}

    # 临时替换配置
    original_configs = ablation_configs.copy()
    ablation_configs.clear()
    ablation_configs.update(non_full_configs)

    print("Running non-full ablation configurations:")
    for config_name in ablation_configs.keys():
        print(f"  - {config_name}")

    # 运行非full的消融实验
    results = run_ablation_study()

    # 恢复完整配置（可选）
    ablation_configs.clear()
    ablation_configs.update(original_configs)

    print("\n🎉 Non-full ablation study completed!")
