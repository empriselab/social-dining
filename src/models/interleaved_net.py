import os

import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization, add
from keras.layers.pooling import GlobalAvgPool2D
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras.backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import categorical_accuracy
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
import ast

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# param

learning_rate = float(sys.argv[1])
batch_size = int(sys.argv[2])
decay_rate = float(sys.argv[3])
l2_value = float(sys.argv[4])
epoch = int(sys.argv[5])
'''

# param
learning_rate = 0.001
batch_size = 128
decay_rate = 0
l2_value = 0
epoch = 10000
'''


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def shuffle_train_test_split(size, ratio):
    index = np.arange(size)
    shuffle(index, random_state=0)
    sep = int(size * ratio)
    return index[:sep], index[sep:]


# read data
'''
time_series = np.load("data/main_np.npy", allow_pickle=True)
subject2 = np.load("data/dinner1_np.npy", allow_pickle=True)
subject3 = np.load("data/dinner2_np.npy", allow_pickle=True)
'''
time_series = np.load("data/target_np.npy", allow_pickle=True)
subject2 = np.load("data/right_np.npy", allow_pickle=True)
subject3 = np.load("data/left_np.npy", allow_pickle=True)
i3d = np.load("data/i3d_np.npy", allow_pickle=True)


X = np.nan_to_num(time_series)
subject2 = np.nan_to_num(subject2)
subject3 = np.nan_to_num(subject3)

assert not np.any(np.isnan(X))
assert not np.any(np.isnan(subject2))
assert not np.any(np.isnan(subject3))

y = np.append(np.ones(6830), np.zeros(2486))

counts = np.bincount(y.astype(int))

y = y.reshape((-1, 1)).T.flatten()

print(len(X))
print(len(y))


# normalization
min_max_scaler = preprocessing.MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X.reshape(-1, 256))
X_norm = X_norm.reshape(-1, 180 * 256)

min_max_scaler = preprocessing.MinMaxScaler()
X2_norm = min_max_scaler.fit_transform(subject2.reshape(-1, 256))
X2_norm = X2_norm.reshape(-1, 180 * 256)

min_max_scaler = preprocessing.MinMaxScaler()
X3_norm = min_max_scaler.fit_transform(subject3.reshape(-1, 256))
X3_norm = X3_norm.reshape(-1, 180 * 256)

i3d = i3d.reshape(len(X), 1024)

X = np.concatenate((X_norm, X2_norm, X3_norm, i3d), axis=1)
# shuffle and split
# train_index, test_index = shuffle_train_test_split(len(y), 0.8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''
rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_resample(X_train, y_train)
'''
i3d_train = X_train[:, 3*180*256:].reshape((-1, 1024))
i3d_test = X_test[:, 3*180*256:].reshape((-1, 1024))
X_train = X_train[:, :3*180*256].reshape((-1, 3*180, 256))
X_test = X_test[:, :3*180*256].reshape((-1, 3*180, 256))

X1_train = X_train[:, :180, :]
X2_train = X_train[:, 180: 2*180, :]
X3_train = X_train[:, 2*180:3*180, :]
X1_test = X_test[:, :180, :]
X2_test = X_test[:, 180: 2*180, :]
X3_test = X_test[:, 2*180:3*180, :]

#y_train = pd.get_dummies(y_train, columns=['l1', 'l2'])
#y_test = pd.get_dummies(y_test, columns=['l1', 'l2'])



assert not np.any(np.isnan(X1_train))
assert not np.any(np.isnan(y_train))

print("Training")
print(X1_train.shape)
print(X2_train.shape)
print(X3_train.shape)
print(i3d_train.shape)
print(y_train.shape)

print("Test")
print(X1_test.shape)
print(X2_test.shape)
print(X3_test.shape)
print(i3d_test.shape)
print(y_test.shape)


print(X1_train[30, :10, :10])
print(X2_train[300, 100:120, 250:])
print(X3_train[3000, 150:160, 250:])

# open pose channel
n_body_pose = 14 * 2
n_hands_pose = 21 * 4
n_face_pose = 70 * 2

n_gaze_pose = 1 * 2
n_head_pose = 1 * 2

n_time = 1

n_speaking = 1

input1 = Input(shape=(180, n_body_pose + n_hands_pose + n_face_pose + n_gaze_pose + n_head_pose + n_time + n_speaking, 1))
bn11 = BatchNormalization()(input1)
conv11 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    bn11)
bn12 = BatchNormalization()(conv11)
conv12 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    bn12)
bn13 = BatchNormalization()(conv12)

conv13 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    bn13)
bn14 = BatchNormalization()(conv13)

conv14 = Conv2D(16, kernel_size=3, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    bn14)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv14)
bn15 = BatchNormalization()(pool11)
conv15 = Conv2D(8, kernel_size=1, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    bn15)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv15)
bn16 = BatchNormalization()(pool12)
flat1 = Flatten()(bn16)


input2 = Input(shape=(180, n_body_pose + n_hands_pose + n_face_pose + n_gaze_pose + n_head_pose + n_time + n_speaking, 1))
bn21 = BatchNormalization()(input2)
add21 = add([bn12, bn14])
conv21 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    add21)
bn22 = BatchNormalization()(conv21)
add22 = add([bn13, bn22])
conv22 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    add22)
bn23 = BatchNormalization()(conv22)
add23 = add([bn23, bn14])
conv23 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    add23)
bn24 = BatchNormalization()(conv23)

conv24 = Conv2D(16, kernel_size=3, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    bn24)
pool21 = MaxPooling2D(pool_size=(2, 2))(bn24)
bn25 = BatchNormalization()(pool21)
conv25 = Conv2D(8, kernel_size=1, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    bn25)
pool22 = MaxPooling2D(pool_size=(2, 2))(bn25)
bn26 = BatchNormalization()(pool22)
flat2 = Flatten()(bn26)

input3 = Input(shape=(180, n_body_pose + n_hands_pose + n_face_pose + n_gaze_pose + n_head_pose + n_time + n_speaking, 1))
bn31 = BatchNormalization()(input3)
add31 = add([bn22, bn24])
conv31 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    add31)
bn32 = BatchNormalization()(conv31)
add32 = add([bn23, bn32])
conv32 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    add32)
bn33 = BatchNormalization()(conv32)
add33 = add([bn33, bn24])
conv33 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    add33)
bn34 = BatchNormalization()(conv33)
conv34 = Conv2D(16, kernel_size=3, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    bn34)
pool31 = MaxPooling2D(pool_size=(2, 2))(bn34)
bn35 = BatchNormalization()(pool31)
conv35 = Conv2D(16, kernel_size=1, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
    bn35)
pool32 = MaxPooling2D(pool_size=(2, 2))(bn35)
bn36 = BatchNormalization()(pool32)
flat3 = Flatten()(bn36)

i3d_inception_dimension = 1024
input4 = Input(shape=(i3d_inception_dimension,))
bn4 = BatchNormalization()(input4)

merge = concatenate([flat1, flat2, flat3])

hidden1 = Dense(32, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(merge)
dropout1 = Dropout(0.05)(hidden1)
hidden2 = Dense(8, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(dropout1)
dropout2 = Dropout(0.05)(hidden2)
bn5 = BatchNormalization()(dropout2)

output = Dense(1, activation='sigmoid')(bn5)


model = Model([input1, input2, input3], output)
model.summary()

metrics = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

TRAIN = True

if TRAIN is True:
    weight_for_0 = 1 / counts[0]
    weight_for_1 = 1 / counts[1]
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print("weights: ", class_weight)

    # learning rate decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_rate)

    # early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

    # save the best model by measuring F1-score
    mc = ModelCheckpoint("checkpoints/interleaved_net_6s_he_visual_time_audio" + str(
        learning_rate) + "_" + str(decay_rate) + "_" + str(l2_value) + '_' + str(batch_size) + ".h5",
                         monitor='val_get_f1', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=binary_crossentropy, metrics=[get_f1, metrics])

    history = model.fit(x=[X1_train, X2_train, X3_train], y=y_train, epochs=epoch,
                        batch_size=batch_size, validation_data=([X1_test, X2_test, X3_test], y_test), callbacks=[es, mc])

else:
    trained_model = keras.models.load_model("checkpoints/best_person1_32_32_0.0001_0_0.1_128.h5",
                                            custom_objects={'get_f1': get_f1})
    predict_layer = Model(inputs=trained_model.input, outputs=trained_model.output)
    predict_layer_output = predict_layer.predict(x=[X1_test, X2_test, X3_test])
    print(predict_layer_output)

    predicted_y = np.argmax(predict_layer_output, axis=1)
    # predicted_y = pd.get_dummies(predicted_y)
    print(predicted_y)

    from sklearn.metrics import f1_score

    print(get_f1(y[test_index], predicted_y))
