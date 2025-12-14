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
config.gpu_options.allow_growth = True  # 程序按需申请内存
sess = tf.compat.v1.Session(config=config)

version = "level8_80%_1"
model_path = "simple" + version
model_name = "simple " + version + ".h5"

max_traj_len = 81
extra_dim = 9
dp_dim = 3
level = 8
lstm_units = 512
dense_extra_units = 18
dense_concat_units = 1024

is_dp = True


class attention(Layer):

    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences
        })
        return config


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
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                      (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
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


def build_model(dp):
    input_gps = Input(shape=(max_traj_len, 2), name='input_gps')
    input_extra = Input(shape=(extra_dim,), name='input_extra')
    input_dp = Input(shape=(dp_dim,), name='input_dp')
    if dp:
        forward_lstm_gps = ConditionalRecurrent(
            LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        backward_lstm_gps = ConditionalRecurrent(
            LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01),
                 go_backwards=True))
        bilstm_inputs = (input_gps, input_dp)
    else:
        forward_lstm_gps = LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01))
        backward_lstm_gps = LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                 go_backwards=True)
        bilstm_inputs = input_gps
    bilstm_gps = Bidirectional(layer=forward_lstm_gps, backward_layer=backward_lstm_gps)(inputs=bilstm_inputs)
    attention_gps = attention2(max_traj_len)(bilstm_gps)
    dense_extra = Dense(dense_extra_units, activation='relu')(input_extra)
    concat = Concatenate()([attention_gps, dense_extra])
    dense_concat = Dense(dense_concat_units, activation='relu')(concat)
    dropout_concat = Dropout(0.5)(dense_concat)
    softmax_concat = Dense(965, activation='softmax')(dropout_concat)

    model = Model(inputs=[input_gps, input_dp, input_extra], outputs=softmax_concat)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5),
                           tfa.metrics.F1Score(num_classes=965, average='macro')])
    return model


def load_model(path, custom_objects=None):
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)
    return model


def normalize(data, level):
    for i in range(len(data)):
        data[i] = [[k[0] / pow(2, level), k[1] / pow(2, level)] for k in data[i]]
    return data


def coords_padding(data, maxlen):
    for i in range(len(data)):
        if len(data[i]) < maxlen:
            data[i] = data[i] + [[0, 0]] * (maxlen - len(data[i]))
    return data


def coords_padding(data, maxlen):
    padded_data = []
    for i in range(len(data)):
        traj = data[i]
        if len(traj) < maxlen:
            traj = traj + [[0, 0]] * (maxlen - len(traj))
        elif len(traj) > maxlen:
            traj = traj[:maxlen]
        # 确保每个元素都是长度为2的列表
        traj = [[item[0], item[1]] if len(item) == 2 else [0, 0] for item in traj]
        padded_data.append(traj)
    return padded_data


def padding(data):
    # max_ori_traj_len == 91(level 8) 81(level 7) 75(level 6) 65(level 5)
    return sequence.pad_sequences(data, padding='post', maxlen=max_traj_len)


def to_one_hot(data):
    return to_categorical(data, num_classes=965)


def calculate_top_k_accuracy(logits, targets, k):
    values, indices = tf.math.top_k(logits, k=k, sorted=True)
    y = tf.cast(tf.reshape(targets, [-1, 1]), tf.int32)
    correct = tf.cast(tf.equal(y, indices), tf.float32)
    top_k_accuracy = tf.reduce_mean(correct) * k
    return top_k_accuracy


def top_k_dispersion(logits, k, center_clusters):
    # last for approx 9 minutes
    values, indices = tf.math.top_k(logits, k=k, sorted=True)
    all_dist = []
    for i in range(len(indices)):
        # print(i)
        tmp_dist = []
        for gps1, gps2 in combinations(indices[i], 2):
            gps1_lon = radians(center_clusters.iloc[int(gps1)]['center_lon'])
            gps1_lat = radians(center_clusters.iloc[int(gps1)]['center_lat'])
            gps2_lon = radians(center_clusters.iloc[int(gps2)]['center_lon'])
            gps2_lat = radians(center_clusters.iloc[int(gps2)]['center_lat'])
            tmp_dist.append(
                (haversine_distances([[gps1_lat, gps1_lon], [gps2_lat, gps2_lon]]) * 6371000 / 1000)[0][1] * 1000)
        all_dist.append(tmp_dist)
    all_dist = np.array(all_dist)
    return np.mean(all_dist), np.mean(np.std(all_dist, axis=1)), np.mean(np.max(all_dist, axis=1))


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


def train(train_per=None):
    train_data, val_data = load_train_data(train_per)

    # train_x = [eval(i) for i in train_data['grids_of_traj']]
    train_x_gps = [eval(i) for i in train_data['coords_of_traj']]
    train_y = np.array(to_one_hot(train_data['dest_cluster']))
    # val_x = [eval(i) for i in val_data['grids_of_traj']]
    val_x_gps = [eval(i) for i in val_data['coords_of_traj']]
    val_y = np.array(to_one_hot(val_data['dest_cluster']))

    train_x_gps = normalize(train_x_gps, level)
    val_x_gps = normalize(val_x_gps, level)
    train_x_gps = np.array(coords_padding(train_x_gps, max_traj_len))
    val_x_gps = np.array(coords_padding(val_x_gps, max_traj_len))
    # train_x = np.array(padding(train_x))
    # val_x = np.array(padding(val_x))

    train_x_extra = np.array(train_data[['daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4',
                                         'weather_5', 'hour_sin', 'hour_cos']])
    # train_x_extra = np.array(train_data[['daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4', 'weather_5', 'hour_sin', 'hour_cos', 'dp_cur', 'dp_30min_prev']])
    val_x_extra = np.array(val_data[
                               ['daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4', 'weather_5',
                                'hour_sin', 'hour_cos']])
    # val_x_extra = np.array(val_data[['daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4', 'weather_5', 'hour_sin', 'hour_cos', 'dp_cur', 'dp_30min_prev']])

    train_x_dp = np.array(train_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])
    val_x_dp = np.array(val_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])

    # sequential model
    # model = tf.keras.Sequential([
    #     # Embedding(16385, 72, input_length = 72, mask_zero = True),
    #     Masking(mask_value = 0, input_shape = (72, 2)),
    #     Bidirectional(LSTM(256, return_sequences = True, kernel_regularizer=tf.keras.regularizers.L2(0.01))),
    #     attention2(72),
    #     Dense(512, activation='relu'),
    #     Dropout(0.5),
    #     Dense(965, activation='softmax')
    # ])

    # model structure
    input_gps = Input(shape=(max_traj_len, 2), name='input_gps')
    input_extra = Input(shape=(extra_dim,), name='input_extra')
    input_dp = Input(shape=(dp_dim,), name='input_dp')

    # masking_gps = Masking(mask_value = 0)(input_gps)
    # bilstm_gps = Bidirectional(LSTM(256, return_sequences = True, kernel_regularizer=tf.keras.regularizers.L2(0.01)))(masking_gps)
    if is_dp:
        forward_lstm_gps = ConditionalRecurrent(
            LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        backward_lstm_gps = ConditionalRecurrent(
            LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01),
                 go_backwards=True))
        bilstm_inputs = (input_gps, input_dp)
    else:
        forward_lstm_gps = LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01))
        backward_lstm_gps = LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                 go_backwards=True)
        bilstm_inputs = input_gps

    bilstm_gps = Bidirectional(layer=forward_lstm_gps, backward_layer=backward_lstm_gps)(inputs=bilstm_inputs)
    attention_gps = attention2(max_traj_len)(bilstm_gps)

    dense_extra = Dense(dense_extra_units, activation='relu')(input_extra)

    concat = Concatenate()([attention_gps, dense_extra])
    dense_concat = Dense(dense_concat_units, activation='relu')(concat)
    dropout_concat = Dropout(0.5)(dense_concat)
    softmax_concat = Dense(965, activation='softmax')(dropout_concat)

    model = Model(inputs=[input_gps, input_dp, input_extra], outputs=softmax_concat)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5),
                           tfa.metrics.F1Score(num_classes=965, average='macro')])
    model.summary()

    checkpoint_save_path = "simple/" + version + "/checkpoint"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        monitor='val_top_k_categorical_accuracy',
        mode='max',
        save_best_only=True)

    log_dir = "../statistic/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # 每个 epoch 记录一次权重直方图
        write_graph=True,  # 记录计算图
        update_freq='epoch'  # 按 epoch 记录，也可以改为 'batch'
    )

    history = model.fit([train_x_gps, train_x_dp, train_x_extra], train_y, epochs=200, batch_size=128,
                        validation_data=([val_x_gps, val_x_dp, val_x_extra], val_y),
                        callbacks=[model_checkpoint_callback, tensorboard_callback])

    model.load_weights(checkpoint_save_path)
    model.save(model_path + "/" + model_name)

    # predict(model)

    # plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.ylim([1, 5])
    # plt.legend(loc='lower right')
    # plt.show()


def batch_train():
    per = ["70%", "90%"]
    # per = ["80%", "70%"]
    for p in per:
        global version
        global model_path
        global model_name
        version = "level8_" + p + "_1"
        model_path = "simple/" + version
        model_name = "simple " + version + ".h5"
        train(p)


def predict(model=None):
    print_content = ""
    # build model
    input_gps = Input(shape=(max_traj_len, 2), name='input_gps')
    input_extra = Input(shape=(extra_dim,), name='input_extra')
    input_dp = Input(shape=(dp_dim,), name='input_dp')
    if is_dp:
        forward_lstm_gps = ConditionalRecurrent(
            LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        backward_lstm_gps = ConditionalRecurrent(
            LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01),
                 go_backwards=True))
        bilstm_inputs = (input_gps, input_dp)
    else:
        forward_lstm_gps = LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01))
        backward_lstm_gps = LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                 go_backwards=True)
        bilstm_inputs = input_gps
    bilstm_gps = Bidirectional(layer=forward_lstm_gps, backward_layer=backward_lstm_gps)(inputs=bilstm_inputs)
    attention_gps = attention2(max_traj_len)(bilstm_gps)
    dense_extra = Dense(dense_extra_units, activation='relu')(input_extra)
    concat = Concatenate()([attention_gps, dense_extra])
    dense_concat = Dense(dense_concat_units, activation='relu')(concat)
    dropout_concat = Dropout(0.5)(dense_concat)
    softmax_concat = Dense(965, activation='softmax')(dropout_concat)

    model = Model(inputs=[input_gps, input_dp, input_extra], outputs=softmax_concat)
    # model = Model(inputs=[input_gps, input_extra], outputs=softmax_concat)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5),
                           tfa.metrics.F1Score(num_classes=965, average='macro')])
    model.load_weights("simple/level8_80%_1/checkpoint")

    # load model
    # model = load_model(model_path + "/" + model_name, custom_objects={'attention2': attention2, 'ConditionalRecurrent': ConditionalRecurrent})
    # model = load_model("simple/trajs only/90%/256 BiLSTM-512 Dense/simple 2.0 coords new.h5", custom_objects={'attention2': attention2})
    # model = load_model("model/simple/实验一/level8_80%/simple level8_80%.h5", custom_objects={'attention2': attention2, 'ConditionalRecurrent': ConditionalRecurrent})
    # ----------------------------------------------------------------------------------------------------------------------------------------------#
    # load data
    for per in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        # for per in ['8']:
        print_content += "--------------------" + per + "0%--------------------\n"
        test_data = load_test_data(per)
        # test_data = test_data[test_data['daytype'] == 1]
        # test_data['hour'] = test_data['trip_start_time'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f").hour)
        # test_data = test_data[(test_data['hour'] >= 7) & (test_data['hour'] <= 9) | (test_data['hour'] >= 17) & (test_data['hour'] <= 19)]
        # test_data = test_data[(test_data['hour'] >= 10) & (test_data['hour'] <= 16) | (test_data['hour'] >= 20) | (test_data['hour'] <= 6)]
        # test_data.reset_index(drop=True, inplace=True)

        # test_x = [eval(i) for i in test_data['grids_of_traj']]
        test_x_gps = [eval(i) for i in test_data['coords_of_traj']]
        test_y_one_hot = np.array(to_one_hot(test_data['dest_cluster']))

        test_y = np.array(test_data['dest_cluster'])

        # test_x = np.array(padding(test_x))

        test_x_gps = normalize(test_x_gps, level)
        test_x_gps = np.array(coords_padding(test_x_gps, max_traj_len))

        test_x_extra = np.array(test_data[['daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4',
                                           'weather_5', 'hour_sin', 'hour_cos']])
        # test_x_dp = np.array(test_data[['dp_cur', 'dp_30min_prev']])
        test_x_dp = np.array(test_data[['dp_cur', 'dp_30min_prev', 'dp_1h_prev']])
        # ----------------------------------------------------------------------------------------------------------------------------------------------#
        # model evaluation(accuracy & loss)
        y_test_pred = model.predict([test_x_gps, test_x_dp, test_x_extra], batch_size=128, verbose=1)
        # y_test_pred = model.predict([test_x_gps, test_x_extra], batch_size=128, verbose=1)

        # loss, accuracy = model.evaluate([test_x_gps, test_x_extra], test_y, batch_size=128, verbose=1)
        # print("Test evaluation: ")
        # print("loss: ", loss)
        # print("accuracy: ", accuracy)
        # ----------------------------------------------------------------------------------------------------------------------------------------------#
        # Top-k Accuracy
        # for i in range(1, 6):
        #     m = tf.keras.metrics.TopKCategoricalAccuracy(k=i)
        #     m.update_state(test_y_one_hot, y_test_pred)
        #     print("Top-K Accuracy:(k={0}) {1}".format(i, m.result().numpy()))
        #     print_content += "Top-K Accuracy:(k={0}) {1}\n".format(i, m.result().numpy())
        # ----------------------------------------------------------------------------------------------------------------------------------------------#
        # Top-1 Avg Distance
        # y_test_pred_argmax = np.argmax(y_test_pred, axis=1)
        center_clusters_gps = pd.read_csv(
            "../processed_data/clusters_5_ring/mean_shift clustering/965_clusters_center_coords.csv")
        # avg_dist = []
        # for i in range(len(test_y)):
        #     y_true_lon = radians(center_clusters_gps.iloc[test_y[i]]['center_lon'])
        #     y_true_lat = radians(center_clusters_gps.iloc[test_y[i]]['center_lat'])
        #     y_pred_lon = radians(center_clusters_gps.iloc[y_test_pred_argmax[i]]['center_lon'])
        #     y_pred_lat = radians(center_clusters_gps.iloc[y_test_pred_argmax[i]]['center_lat'])
        #     tmp_dist = haversine_distances([[y_true_lat, y_true_lon], [y_pred_lat, y_pred_lon]]) * 6371000 / 1000
        #     avg_dist.append(tmp_dist[0][1])
        # print("Top-1 avg_dist(m): ", np.mean(avg_dist) * 1000)
        # print_content += "Top-1 avg_dist(m): " + str(np.mean(avg_dist) * 1000) + "\n"
        # print("Top-1 median_dist(m): ", np.median(avg_dist) * 1000)
        # print_content += "Top-1 median_dist(m): " + str(np.median(avg_dist) * 1000) + "\n"
        # print("Top-1 RMSE_dist: ", np.sqrt(np.mean(np.square(avg_dist))))
        # print_content += "Top-1 RMSE_dist: " + str(np.sqrt(np.mean(np.square(avg_dist)))) + "\n"
        # ----------------------------------------------------------------------------------------------------------------------------------------------#
        # Top-K Dispersion
        # mean, avg_std, avg_max = top_k_dispersion(y_test_pred, k = 5, center_clusters=center_clusters_gps)
        # print("mean: ", mean)
        # print_content += "mean: " + str(mean) + "\n"
        # print("avg_std: ", avg_std)
        # print_content += "avg_std: " + str(avg_std) + "\n"
        # print("avg_max: ", avg_max)
        # print_content += "avg_max: " + str(avg_max) + "\n\n"
        # ----------------------------------------------------------------------------------------------------------------------------------------------#
        # Top-K Avg Distance
        K = [1, 5, 10]
        for k in K:
            mean, avg_median, avg_std = top_k_avg_dist(y_test_pred, test_y, k, center_clusters=center_clusters_gps)
            print("Top-{0} Distance mean: {1}".format(k, mean))
            print_content += "Top-" + str(k) + " Distance mean: " + str(mean) + "\n"
            # print("Top-K Distance avg_median: ", avg_median)
            # print_content += "Top-K Distance avg_median: " + str(avg_median) + "\n"
            # print("Top-K Distance avg_std: ", avg_std)
            # print_content += "Top-K Distance avg_std: " + str(avg_std) + "\n\n"

    with open(model_path + "/" + "log.txt", "w") as f:
        f.write(print_content)


def case_study(case_no):
    # build model
    if is_dp:
        model = build_model(dp=True)
        # model.load_weights("simple/前置实验组/level8_80%_dp/checkpoint")
        model.load_weights("simple/level8_80%_0.5h_1/checkpoint")
    else:
        model = build_model(dp=False)
        model.load_weights("simple/level8_80%_1/checkpoint")

    # load data
    # case_no = 5300
    center_clusters_gps = pd.read_csv(
        "../processed_data/clusters_5_ring/mean_shift clustering/965_clusters_center_coords.csv")
    for per in ['8']:
        test_data = load_test_data(per)
        # test_data = test_data[test_data['daytype'] == 1]
        test_x_gps = [eval(test_data['coords_of_traj'].iloc[case_no])]
        test_y_one_hot = np.array(to_one_hot(test_data['dest_cluster'].iloc[case_no]))
        test_y = np.array(test_data['dest_cluster'].iloc[case_no])
        print("length: ", len(test_x_gps[0]))
        test_x_gps = normalize(test_x_gps, level)
        test_x_gps = np.array(coords_padding(test_x_gps, max_traj_len))
        test_x_extra = np.array(test_data[['daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4',
                                           'weather_5', 'hour_sin', 'hour_cos']].iloc[case_no]).reshape(1, -1)
        test_x_dp = np.array(test_data[['dp_cur', 'dp_30min_prev']].iloc[case_no]).reshape(1, -1)
        # 'dp_1h_prev'
        # predict
        y_test_pred = model.predict([test_x_gps, test_x_dp, test_x_extra], batch_size=1, verbose=1)

        values, indices = tf.math.top_k(y_test_pred, k=5, sorted=True)

        print(per + "0%")
        print("True Destination: ", test_y)
        print("Predicted Destination:")
        for i in range(len(indices[0])):
            y_true_lon = radians(center_clusters_gps.iloc[test_y]['center_lon'])
            y_true_lat = radians(center_clusters_gps.iloc[test_y]['center_lat'])
            y_pred_lon = radians(center_clusters_gps.iloc[int(indices[0][i])]['center_lon'])
            y_pred_lat = radians(center_clusters_gps.iloc[int(indices[0][i])]['center_lat'])
            tmp_dist = haversine_distances([[y_true_lat, y_true_lon], [y_pred_lat, y_pred_lon]]) * 6371000 / 1000
            print("NO:{0}   Prob:{1}   Dist:{2}".format(int(indices[0][i]), float(values[0][i]), tmp_dist[0][1] * 1000))


def case_study2():
    c = set()

    # build model
    model = build_model(dp=False)
    model.load_weights("simple/level8_80%_1/checkpoint")
    model_dp = build_model(dp=True)
    model_dp.load_weights("simple/level8_80%_0.5h_1/checkpoint")

    # predict and compare
    for per in ['8']:
        test_data = load_test_data(per)
        # test_data = test_data[test_data['daytype'] == 1]

        test_x_gps = [eval(i) for i in test_data['coords_of_traj']]
        test_y = np.array(test_data['dest_cluster'])

        test_x_gps = normalize(test_x_gps, level)
        test_x_gps = np.array(coords_padding(test_x_gps, max_traj_len))

        test_x_extra = np.array(test_data[['daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4',
                                           'weather_5', 'hour_sin', 'hour_cos']])
        test_x_dp = np.array(test_data[['dp_cur', 'dp_30min_prev']])

        y_test_pred = model.predict([test_x_gps, test_x_dp, test_x_extra], batch_size=128, verbose=1)
        y_test_pred_dp = model_dp.predict([test_x_gps, test_x_dp, test_x_extra], batch_size=128, verbose=1)

        indices = tf.math.top_k(y_test_pred, k=5, sorted=True)[1]
        indices_dp = tf.math.top_k(y_test_pred_dp, k=5, sorted=True)[1]

        ori_clusters = [708, 760, 761, 768, 769, 770, 771, 781, 782, 802, 803, 804, 805, 806, 807, 808, 809, 867, 868,
                        869, 871, 872, 873, 874, 875]

        for i in range(len(test_data)):
            if test_y[i] in indices_dp[i] and test_y[i] not in indices[i]:
                h = datetime.strptime(test_data['trip_start_time'].iloc[i], "%Y-%m-%d %H:%M:%S.%f").hour
                if h >= 7 and h <= 9:
                    if test_data['ori_cluster'].iloc[i] in ori_clusters:
                        c.add(i)
    print(c)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--arg1', help='argument 1')
    # parser.add_argument('-dp', '--arg2', help='argument 1')
    # args = parser.parse_args()
    # if args.arg2 == "1":
    #     is_dp = True
    # else:
    #     is_dp = False
    # batch_train()
    # train("80%")
    predict()
    # case_study(int(args.arg1))
    # case_study2()

