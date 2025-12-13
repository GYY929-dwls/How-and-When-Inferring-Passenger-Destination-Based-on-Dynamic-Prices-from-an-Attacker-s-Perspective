# ========== 基于Transformer的轨迹目的地预测模型 ==========

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from math import radians
import os
import pickle
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics.pairwise import haversine_distances

# GPU配置
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# 模型配置
version = "transformer_final_v1"
model_path = "transformer/" + version
model_name = f"transformer_{version}.h5"

# 超参数
max_traj_len = 81
per_step_feat_dim = 2
extra_dim = 9
dp_dim = 3

# Transformer架构参数
d_model = 256
num_heads = 8
num_layers = 3
dff = 896
dense_extra_units = 32
dense_concat_units = 1024
dropout_rate = 0.12
label_smoothing = 0.1
num_classes = 965
is_dp = True
level = 8


def haversine_distance_meters(lon1, lat1, lon2, lat2):
    """计算哈夫辛距离（米）"""
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi, dlambda = radians(lat2 - lat1), radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def prepare_trajectory_features(trajectories, maxlen=max_traj_len):
    """预处理轨迹特征：归一化与填充"""
    processed_features = []

    for traj in trajectories:
        if not isinstance(traj, list) or len(traj) == 0:
            traj = [[0.0, 0.0]]

        traj = traj[:maxlen]

        # 归一化处理
        normalized_traj = [
            [lon / (2 ** level), lat / (2 ** level)]
            for lon, lat in traj
        ]

        # 序列填充
        if len(normalized_traj) < maxlen:
            padding_length = maxlen - len(normalized_traj)
            normalized_traj.extend([[0.0, 0.0] for _ in range(padding_length)])

        processed_features.append(normalized_traj)

    return np.array(processed_features, dtype=np.float32)


def create_attention_mask(features):
    """创建注意力掩码"""
    features_tensor = tf.convert_to_tensor(features)
    is_padding = tf.reduce_all(tf.math.equal(features_tensor, 0.0), axis=-1)
    return tf.logical_not(is_padding)


def truncate_trajectory_by_prefix_ratio(trajectory, prefix_ratio):
    """按比例截断轨迹"""
    if not isinstance(trajectory, list) or len(trajectory) == 0:
        return [[0.0, 0.0]]

    prefix_len = max(1, int(np.ceil(len(trajectory) * prefix_ratio)))
    return trajectory[:prefix_len]


def compute_top1_distance_error(predictions, ground_truth, cluster_centers):
    """计算Top-1预测的平均距离误差"""
    predicted_indices = np.argmax(predictions, axis=1)
    distance_errors = []

    for i, pred_idx in enumerate(predicted_indices):
        pred_lon = radians(cluster_centers.iloc[int(pred_idx)]['center_lon'])
        pred_lat = radians(cluster_centers.iloc[int(pred_idx)]['center_lat'])
        true_lon = radians(cluster_centers.iloc[int(ground_truth[i])]['center_lon'])
        true_lat = radians(cluster_centers.iloc[int(ground_truth[i])]['center_lat'])

        distance = haversine_distances(
            [[pred_lat, pred_lon], [true_lat, true_lon]]
        ) * 6371000
        distance_errors.append(distance[0][1])

    return np.mean(distance_errors)


class SinusoidalPositionalEncoding(Layer):
    """正弦位置编码层"""

    def __init__(self, max_length, d_model):
        super().__init__()
        self.positional_encoding = self._generate_positional_encoding(max_length, d_model)

    def _get_frequency_angles(self, position, index, d_model):
        angle_rate = 1 / np.power(10000, (2 * (index // 2)) / np.float32(d_model))
        return position * angle_rate

    def _generate_positional_encoding(self, max_length, d_model):
        angle_radians = self._get_frequency_angles(
            np.arange(max_length)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        angle_radians[:, 0::2] = np.sin(angle_radians[:, 0::2])
        angle_radians[:, 1::2] = np.cos(angle_radians[:, 1::2])

        return tf.cast(angle_radians[np.newaxis, ...], tf.float32)

    def call(self, inputs, **kwargs):
        sequence_length = tf.shape(inputs)[1]
        return inputs + self.positional_encoding[:, :sequence_length, :]


class TransformerEncoderLayer(Layer):
    """Transformer编码器单层"""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )

        self.feed_forward_network = tf.keras.Sequential([
            Dense(dff, activation='gelu'),
            Dense(d_model)
        ])

        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training, mask=None):
        attention_output = self.multi_head_attention(
            inputs, inputs, attention_mask=mask
        )
        attention_output = self.dropout1(attention_output, training=training)
        output1 = self.layer_norm1(inputs + attention_output)

        ffn_output = self.feed_forward_network(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layer_norm2(output1 + ffn_output)

        return output2


class SpatioTemporalTransformer(Layer):
    """时空Transformer编码器"""

    def __init__(self, num_layers, d_model, num_heads, dff, max_len, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.input_projection = Dense(d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(max_len + 1, d_model)

        self.cls_token = self.add_weight(
            name="classification_token",
            shape=(1, 1, d_model),
            initializer='glorot_uniform',
            trainable=True
        )

        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs)[0]

        projected_inputs = self.input_projection(inputs) * tf.math.sqrt(
            tf.cast(self.d_model, tf.float32)
        )

        encoded_inputs = self.positional_encoding(projected_inputs)

        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        transformer_inputs = tf.concat([cls_tokens, encoded_inputs], axis=1)

        if mask is not None:
            if mask.dtype != tf.bool:
                mask = tf.cast(mask, tf.bool)
            cls_mask = tf.ones((batch_size, 1), dtype=tf.bool)
            extended_mask = tf.concat([cls_mask, mask], axis=1)
            extended_mask = tf.expand_dims(extended_mask, 1)
            mask = extended_mask

        transformer_inputs = self.dropout(transformer_inputs, training=training)

        encoder_outputs = transformer_inputs
        for encoder_layer in self.encoder_layers:
            encoder_outputs = encoder_layer(
                encoder_outputs, training=training, mask=mask
            )

        return encoder_outputs


class ConditionalTransformer(Layer):
    """条件Transformer（支持深度特征调制）"""

    def __init__(self, num_layers, d_model, num_heads, dff, max_len, dp_dim, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.max_len = max_len
        self.dp_dim = dp_dim
        self.dropout_rate = dropout_rate

        self.depth_feature_processor = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dropout(dropout_rate * 0.5),
            Dense(d_model * 2)
        ])

        self.spatiotemporal_transformer = SpatioTemporalTransformer(
            num_layers, d_model, num_heads, dff, max_len, dropout_rate
        )

        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None, mask=None):
        trajectory_features, depth_features = inputs

        transformer_output = self.spatiotemporal_transformer(
            trajectory_features, training=training, mask=mask
        )

        processed_depth_features = self.depth_feature_processor(depth_features)
        gamma, beta = tf.split(processed_depth_features, 2, axis=-1)

        gamma = tf.expand_dims(gamma, 1)
        beta = tf.expand_dims(beta, 1)

        modulated_output = transformer_output * (1 + gamma) + beta

        return self.layer_norm(modulated_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "max_len": self.max_len,
            "dp_dim": self.dp_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


def build_transformer_model(use_dp=True, learning_rate=1e-3):
    """构建Transformer预测模型"""

    # 输入层定义
    trajectory_input = Input(
        shape=(max_traj_len, per_step_feat_dim),
        name='trajectory_input'
    )
    context_input = Input(shape=(extra_dim,), name='context_input')
    depth_input = Input(shape=(dp_dim,), name='depth_input')

    # 创建注意力掩码
    attention_mask = create_attention_mask(trajectory_input)
    training_mode = K.learning_phase()

    # Transformer主干选择
    if use_dp:
        transformer_core = ConditionalTransformer(
            num_layers, d_model, num_heads, dff, max_traj_len, dp_dim, dropout_rate
        )
        transformer_output = transformer_core(
            (trajectory_input, depth_input), training=training_mode, mask=attention_mask
        )
    else:
        transformer_core = SpatioTemporalTransformer(
            num_layers, d_model, num_heads, dff, max_traj_len, dropout_rate
        )
        transformer_output = transformer_core(
            trajectory_input, training=training_mode, mask=attention_mask
        )

    # 提取CLS token作为轨迹表示
    trajectory_representation = transformer_output[:, 0, :]

    # 上下文特征编码
    encoded_context = Dense(
        dense_extra_units,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(5e-6)
    )(context_input)
    encoded_context = Dropout(0.25)(encoded_context)

    # 特征融合
    fused_features = Concatenate()([trajectory_representation, encoded_context])

    # 多层特征处理
    processed_features = Dense(
        dense_concat_units,
        activation='gelu',
        kernel_regularizer=tf.keras.regularizers.l2(5e-6)
    )(fused_features)
    processed_features = Dropout(0.3)(processed_features)

    refined_features = Dense(
        dense_concat_units // 2,
        activation='gelu',
        kernel_regularizer=tf.keras.regularizers.l2(5e-6)
    )(processed_features)
    refined_features = Dropout(0.3)(refined_features)

    # 输出层
    predictions = Dense(
        num_classes,
        activation='softmax',
        name='destination_prediction',
        kernel_regularizer=tf.keras.regularizers.l2(5e-6)
    )(refined_features)

    # 模型构建
    model = Model(
        inputs=[trajectory_input, depth_input, context_input],
        outputs=predictions
    )

    # 优化器配置
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0
    )

    # 模型编译
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing,
            from_logits=False
        ),
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )

    return model


def load_training_data(data_percentage="80%"):
    """加载训练与验证数据集"""
    train_data = pd.read_csv(
        f"../training data/{data_percentage}/train_{data_percentage}_all_level8_new_clusters_1.csv"
    )
    validation_data = pd.read_csv(
        f"../validation data/{data_percentage}/val_{data_percentage}_all_level8_new_clusters_1.csv"
    )
    return train_data, validation_data


def load_testing_data(data_percentage='8'):
    """加载测试数据集"""
    test_data = pd.read_csv(
        f"../testing data/{data_percentage}0%/test_{data_percentage}0%_all_level8_new_clusters_1.csv"
    )
    return test_data


def convert_to_one_hot(labels):
    """将标签转换为one-hot编码"""
    return to_categorical(labels, num_classes=num_classes)


def execute_training_and_evaluation(training_percentage="80%"):
    """执行模型训练与性能评估"""

    print("=" * 80)
    print(f"模型训练启动 - 训练集比例: {training_percentage}")
    print("=" * 80)
    print("\n模型配置:")
    print(f"  - 隐层维度 (d_model): {d_model}")
    print(f"  - 初始学习率: {1e-3}")
    print(f"  - Dropout率: {dropout_rate}")
    print(f"  - 批次大小: 128")
    print()

    # 数据加载与预处理
    train_data, validation_data = load_training_data(training_percentage)

    print("数据预处理中...")

    # 轨迹数据解析
    training_trajectories = [eval(x) for x in train_data['coords_of_traj']]
    validation_trajectories = [eval(x) for x in validation_data['coords_of_traj']]

    # 特征准备
    train_features = prepare_trajectory_features(training_trajectories)
    validation_features = prepare_trajectory_features(validation_trajectories)

    # 标签编码
    train_labels = np.array(convert_to_one_hot(train_data['dest_cluster']))
    validation_labels = np.array(convert_to_one_hot(validation_data['dest_cluster']))

    # 上下文特征
    train_context_features = np.array(train_data[[
        'daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3',
        'weather_4', 'weather_5', 'hour_sin', 'hour_cos'
    ]])
    validation_context_features = np.array(validation_data[[
        'daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3',
        'weather_4', 'weather_5', 'hour_sin', 'hour_cos'
    ]])

    # 深度特征
    train_depth_features = np.array(train_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])
    validation_depth_features = np.array(validation_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])

    print(f"训练集样本数: {len(train_data):,}")
    print(f"验证集样本数: {len(validation_data):,}")

    # 模型构建
    print("\n构建Transformer模型...")
    model = build_transformer_model(use_dp=is_dp)
    model.summary()

    # 模型保存目录
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # 训练回调函数配置
    checkpoint_path = model_path + "/checkpoint"

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-5,
        verbose=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=35,
        restore_best_weights=True,
        verbose=1
    )

    # 模型训练
    print("\n开始模型训练...")
    print("目标验证准确率阈值: 40.0%")
    print()

    training_history = model.fit(
        [train_features, train_depth_features, train_context_features],
        train_labels,
        validation_data=(
            [validation_features, validation_depth_features, validation_context_features],
            validation_labels
        ),
        epochs=200,
        batch_size=128,
        callbacks=[checkpoint_callback, learning_rate_scheduler, early_stopping_callback],
        verbose=1
    )

    # 加载最佳模型权重
    model.load_weights(checkpoint_path)
    model.save(model_path + "/" + model_name)

    # 验证集性能评估
    print("\n验证集性能评估:")
    validation_loss, validation_accuracy, validation_top5_accuracy = model.evaluate(
        [validation_features, validation_depth_features, validation_context_features],
        validation_labels,
        batch_size=128,
        verbose=0
    )

    print(f"验证准确率: {validation_accuracy:.4f}")
    print(f"验证Top-5准确率: {validation_top5_accuracy:.4f}")
    print(f"验证损失: {validation_loss:.4f}")

    # 性能分析
    if validation_accuracy < 0.38:
        print("\n性能分析: 验证准确率低于38%，可能存在欠拟合")
    elif validation_accuracy < 0.42:
        print("\n性能分析: 验证准确率处于38-42%区间")
    else:
        print("\n性能分析: 验证准确率超过42%，性能良好")

    # 测试阶段
    print("\n" + "=" * 80)
    print("模型测试阶段 - 使用80%测试集")
    print("=" * 80)

    # 加载聚类中心数据
    cluster_centers = pd.read_csv(
        "../processed_data/clusters_5_ring/mean_shift clustering/965_clusters_center_coords.csv"
    )

    # 加载测试数据
    test_data = load_testing_data('8')

    test_trajectories = [eval(x) for x in test_data['coords_of_traj']]
    test_labels = np.array(test_data['dest_cluster'])
    test_context_features = np.array(test_data[[
        'daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3',
        'weather_4', 'weather_5', 'hour_sin', 'hour_cos'
    ]])
    test_depth_features = np.array(test_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])

    print(f"测试集样本数: {len(test_data):,}")

    # 不同前缀比例测试
    prefix_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    distance_errors = []

    print("\n不同轨迹前缀比例下的预测性能:")
    print("-" * 80)
    print(f"{'前缀比例':<12} {'Top-1距离误差 (米)':<20}")
    print("-" * 80)

    for prefix_ratio in prefix_ratios:
        truncated_trajectories = [
            truncate_trajectory_by_prefix_ratio(traj, prefix_ratio)
            for traj in test_trajectories
        ]

        truncated_features = prepare_trajectory_features(truncated_trajectories)

        predictions = model.predict(
            [truncated_features, test_depth_features, test_context_features],
            batch_size=128,
            verbose=0
        )

        mean_distance_error = compute_top1_distance_error(
            predictions, test_labels, cluster_centers
        )
        distance_errors.append(mean_distance_error)

        print(f"{int(prefix_ratio * 100):>3}%         {mean_distance_error:>8.0f}")

    # 平均误差计算（10-60%前缀比例）
    average_error = np.mean(distance_errors[:6])

    print("-" * 80)
    print(f"{'平均(10-60%)':<12} {average_error:>8.0f}")
    print("-" * 80)

    # 模型性能对比
    print("\n" + "=" * 80)
    print("模型性能对比分析")
    print("=" * 80)

    # BiLSTM基准性能
    bilstm_benchmark = [6177, 5540, 3728, 1997, 1216, 948]
    bilstm_average = 3268

    # 性能对比表格
    print(f"\n{'模型':<20} {'10%':<8} {'20%':<8} {'30%':<8} {'40%':<8} {'50%':<8} {'60%':<8} {'平均':<8}")
    print("-" * 80)

    # BiLSTM性能
    print(f"{'BiLSTM':<20}", end="")
    for error in bilstm_benchmark:
        print(f"{error:<8.0f}", end="")
    print(f"{bilstm_average:<8.0f}")

    # Transformer性能
    print(f"{'Transformer':<20}", end="")
    for i in range(6):
        print(f"{distance_errors[i]:<8.0f}", end="")
    print(f"{average_error:<8.0f}")

    # 性能提升计算
    performance_improvements = [
        (bilstm_benchmark[i] - distance_errors[i]) / bilstm_benchmark[i] * 100
        for i in range(6)
    ]
    average_improvement = (bilstm_average - average_error) / bilstm_average * 100

    print(f"{'提升百分比':<20}", end="")
    for improvement in performance_improvements:
        print(f"{improvement:<8.1f}", end="")
    print(f"{average_improvement:<8.1f}")
    print("=" * 80)

    # 结果分析
    if average_improvement > 0:
        print(f"\n结论: Transformer模型相比BiLSTM平均误差降低{average_improvement:.1f}%")
    elif average_improvement > -5:
        print(f"\n结论: Transformer模型与BiLSTM性能相近，差异{average_improvement:.1f}%")
    else:
        print(f"\n结论: Transformer模型相比BiLSTM平均误差增加{abs(average_improvement):.1f}%")

    # 结果保存
    evaluation_results = {
        'prefix_ratios': [f"{int(r * 100)}%" for r in prefix_ratios],
        'distance_errors': distance_errors,
        'average_error': average_error,
        'validation_accuracy': validation_accuracy,
        'validation_top5_accuracy': validation_top5_accuracy,
        'performance_improvement': average_improvement
    }

    with open(model_path + "/evaluation_results.pkl", "wb") as result_file:
        pickle.dump(evaluation_results, result_file)

    print(f"\n评估结果已保存至: {model_path}/evaluation_results.pkl")

    return model, distance_errors


def execute_testing_only():
    """执行仅测试模式"""
    print("\n" + "=" * 80)
    print("模型测试模式")
    print("=" * 80)

    # 模型文件检查
    model_file_path = model_path + "/" + model_name
    if not os.path.exists(model_file_path):
        print(f"错误: 未找到模型文件: {model_file_path}")
        return

    # 模型加载
    print(f"加载预训练模型: {model_file_path}")
    model = build_transformer_model(use_dp=is_dp)
    model.load_weights(model_file_path)
    print("模型加载完成")

    # 测试数据加载
    cluster_centers = pd.read_csv(
        "../processed_data/clusters_5_ring/mean_shift clustering/965_clusters_center_coords.csv"
    )
    test_data = load_testing_data('8')

    test_trajectories = [eval(x) for x in test_data['coords_of_traj']]
    test_labels = np.array(test_data['dest_cluster'])
    test_context_features = np.array(test_data[['daytype', 'weather_0', 'weather_1', 'weather_2',
                                                'weather_3', 'weather_4', 'weather_5', 'hour_sin', 'hour_cos']])
    test_depth_features = np.array(test_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])

    # 前缀比例测试
    prefix_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    distance_errors = []

    print("\n轨迹前缀比例与预测误差:")
    for ratio in prefix_ratios:
        truncated_trajectories = [
            truncate_trajectory_by_prefix_ratio(traj, ratio)
            for traj in test_trajectories
        ]
        truncated_features = prepare_trajectory_features(truncated_trajectories)

        predictions = model.predict(
            [truncated_features, test_depth_features, test_context_features],
            batch_size=128,
            verbose=0
        )

        distance_error = compute_top1_distance_error(predictions, test_labels, cluster_centers)
        distance_errors.append(distance_error)

        print(f"前缀比例 {int(ratio * 100):>3}%: {distance_error:>8.0f} 米")

    average_distance_error = np.mean(distance_errors)
    print(f"\n平均预测误差 (10-60%前缀比例): {average_distance_error:.0f} 米")
    print("=" * 80)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='基于Transformer的轨迹目的地预测模型')
    parser.add_argument(
        '--train-per',
        type=str,
        default=None,
        help='训练数据比例 (例如: 80)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='仅执行测试模式'
    )

    arguments = parser.parse_args()

    # 执行模式判断
    if arguments.test:
        execute_testing_only()
        exit()

    if arguments.train_per is None:
        print("错误: 未指定训练数据比例或测试模式")
        print("使用说明:")
        print("  训练模式: python transformer_model.py --train-per 80")
        print("  测试模式: python transformer_model.py --test")
        exit()

    # 执行训练与评估
    trained_model, test_results = execute_training_and_evaluation(arguments.train_per)
    print("\n模型训练与评估完成")