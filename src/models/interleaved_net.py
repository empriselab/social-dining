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
# from keras.layers.merge import concatenate
from keras.layers import concatenate
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras.backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys

from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import categorical_accuracy
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
import ast

import random
import tqdm

from data_utils import organize_data

from tensorflow.keras.utils import set_random_seed
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# param


# learning_rate = float(sys.argv[1])
# batch_size = int(sys.argv[2])
# decay_rate = float(sys.argv[3])
# l2_value = float(sys.argv[4])
# epoch = int(sys.argv[5])
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


n_body_pose = 14 * 2
n_hands_pose = 21 * 4
n_face_pose = 70 * 2

n_gaze_pose = 1 * 2
n_head_pose = 1 * 2

n_time = 1

n_speaking = 1

feature_sizes = {'body':14*2, 'face':70*2, 'gaze':2, 'headpose':2, 'time_since_last_bite':1, 'time_since_start':1, 'num_bites':1, 'speaking':1}

gpus = tf.config.list_physical_devices('GPU')

print(gpus)

# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)


class PazNet:
    def __init__(self, l2_value=0, frame_length=90, inputs=['body', 'face', 'gaze', 'headpose', 'speaking'], filter_scale=1, use_ssp=0, num_global_feats_repeat=0): 
        # open pose channel
        self.inputs = inputs
        self.l2_value = l2_value
        self.frame_length = frame_length
        self.use_ssp = use_ssp
        self.filter_scale = filter_scale
        self.num_global_feats_repeat = num_global_feats_repeat

            
        self.time_inputs = 0
        self.count_inputs = 0
        self.input_size = 0

        # time input size
        for inp in inputs:
            if inp in ['time_since_last_bite', 'time_since_start']:
                self.time_inputs += feature_sizes[inp]
            # count input size
            elif inp in ['num_bites']:
                self.count_inputs += feature_sizes[inp]
            else:
                self.input_size += feature_sizes[inp]



    def create_model(self):
        l2_value = self.l2_value
        frame_length = self.frame_length
        input_size = self.input_size
        filter_scale = self.filter_scale
        pool_num = 2
        k_size = 3

        # this is for when the inputs are too small
        if input_size <= feature_sizes['gaze'] + feature_sizes['headpose'] + feature_sizes['speaking']:
            pool_num = 1

        if input_size <= feature_sizes['gaze'] + feature_sizes['headpose']:
            k_size = 1

        # input 1
        input1 = Input(shape=(frame_length, input_size, 1))
        bn11 = BatchNormalization()(input1)
        conv11 = Conv2D(32*filter_scale, k_size, padding='same', kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn11)
        bn12 = BatchNormalization()(conv11)
        conv12 = Conv2D(32*filter_scale, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn12)
        bn13 = BatchNormalization()(conv12)

        conv13 = Conv2D(32*filter_scale, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn13)
        bn14 = BatchNormalization()(conv13)

        conv14 = Conv2D(16*filter_scale, k_size, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn14)
        pool11 = MaxPooling2D(pool_size=(pool_num, pool_num))(conv14)
        bn15 = BatchNormalization()(pool11)
        conv15 = Conv2D(8*filter_scale, kernel_size=1, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn15)
        pool12 = MaxPooling2D(pool_size=(pool_num, pool_num))(conv15)
        bn16 = BatchNormalization()(pool12)
        flat1 = Flatten()(bn16)

        # input 2
        input2 = Input(shape=(frame_length, input_size, 1))
        bn21 = BatchNormalization()(input2)
        add21 = add([bn12, bn14, bn21])
        conv21 = Conv2D(32*filter_scale, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add21)
        bn22 = BatchNormalization()(conv21)
        add22 = add([bn13, bn22])
        conv22 = Conv2D(32*filter_scale, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add22)
        bn23 = BatchNormalization()(conv22)
        add23 = add([bn23, bn14])
        conv23 = Conv2D(32*filter_scale, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add23)
        bn24 = BatchNormalization()(conv23)

        conv24 = Conv2D(16*filter_scale, k_size, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn24)
        pool21 = MaxPooling2D(pool_size=(pool_num, pool_num))(bn24)
        bn25 = BatchNormalization()(pool21)
        conv25 = Conv2D(8*filter_scale, kernel_size=1, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn25)
        pool22 = MaxPooling2D(pool_size=(pool_num, pool_num))(conv25)
        bn26 = BatchNormalization()(pool22)
        flat2 = Flatten()(bn26)


        input3 = Input(shape=(frame_length, input_size, 1))

        bn31 = BatchNormalization()(input3)
        add31 = add([bn22, bn24, bn31])
        conv31 = Conv2D(32*filter_scale, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add31)
        bn32 = BatchNormalization()(conv31)
        add32 = add([bn23, bn32])
        conv32 = Conv2D(32*filter_scale, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add32)
        bn33 = BatchNormalization()(conv32)
        add33 = add([bn33, bn24])
        conv33 = Conv2D(32*filter_scale, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add33)
        bn34 = BatchNormalization()(conv33)
        conv34 = Conv2D(16*filter_scale, k_size, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn34)
        pool31 = MaxPooling2D(pool_size=(pool_num, pool_num))(conv34)
        bn35 = BatchNormalization()(pool31)
        conv35 = Conv2D(16*filter_scale, kernel_size=1, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn35)
        pool32 = MaxPooling2D(pool_size=(pool_num, pool_num))(bn35)
        bn36 = BatchNormalization()(pool32)
        flat3 = Flatten()(bn36)

        # input 4
        print(self.num_global_feats_repeat)
        if self.num_global_feats_repeat == 0:
            global_inputs = Input(shape=(self.count_inputs + self.num_global_feats))
        else:
            global_inputs = Input(shape=(self.count_inputs + self.time_inputs)*self.num_global_feats_repeat)


        if self.use_ssp == 2: # we don't concatenate these features for person 1, which is flat 3
            merge = concatenate([flat1, flat2, global_inputs])
        else:
            # merge them together
            merge = concatenate([flat1, flat2, flat3, global_inputs])

        # get output
        hidden1 = Dense(32, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(merge)
        dropout1 = Dropout(0.05)(hidden1)
        hidden2 = Dense(8, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(dropout1)
        dropout2 = Dropout(0.05)(hidden2)
        bn5 = BatchNormalization()(dropout2)

        output = Dense(1, activation='sigmoid')(bn5)


        self.model = Model([input1, input2, input3, global_inputs], output)
        self.model.summary()

        tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=True)


        self.metrics = [
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

    def load(self, path):
        self.model = keras.models.load_model("../../data/models/interleaved_net_6s_he_nohand_15fps_0.001_0.0_0.005_32.h5",
                                                custom_objects={'get_f1': get_f1})
        self.predict_layer = Model(inputs=self.model.input, outputs=self.model.output)


    def predict(self, person1, person2, person3, global_feats):


        predict_layer_output = self.predict_layer.predict(x=[person1, person2, person3, global_feats])
        print(predict_layer_output)

        predicted_y = np.argmax(predict_layer_output, axis=1)
        # predicted_y = pd.get_dummies(predicted_y)
        print(predicted_y)

    '''
        learning_rate = 0.001
        batch_size = 128
        decay_rate = 0
        l2_value = 0
        epoch = 10000
    '''

def generator(main,left,right,labels, batch_size): # Create empty arrays to contain batch of features and labels# batch_features = np.zeros((batch_size, 64, 64, 3))
    batch_labels = np.zeros((batch_size,1)) 
    p1 = np.zeros((batch_size, main.shape[1], main.shape[2]))
    p2 = np.zeros((batch_size, main.shape[1], main.shape[2]))
    p3 = np.zeros((batch_size, main.shape[1], main.shape[2]))
    cur = 0

    idxs = np.arange(len(main))
    np.random.shuffle(idxs)
    # set random shuffle
    main = main[idxs]
    left = left[idxs]
    right = right[idxs]
    labels = labels[idxs]

    while True:
        for i in range(batch_size):
            # choose random index in features
            if cur == len(labels):
                cur = 0
                # and random shuffle
                idxs = np.random.shuffle(idxs)
                main = main[idxs]
                left = left[idxs]
                right = right[idxs]
                labels = labels[idxs]

            p1[i] = main[cur]
            p2[i] = left[cur]
            p3[i] = right[cur]
            batch_labels[i] = labels[cur]


        yield [p1, p2, p3], batch_labels

def train(args):

    features = args.features
    global_features = args.global_features
    epoch = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    l2_value = args.l2_value
    patience = args.patience
    decay_rate = args.decay_rate
    seed = args.seed
    training_split = args.training_split
    use_ssp = args.use_ssp
    frame_length = args.frame_length


    # set seeds
    np.random.seed(seed)
    set_random_seed(seed)

    # clean out the features and global features lists of junk
    if len(features) == 1 and features[0] == 'None':
        features = []
    if len(global_features) == 1 and global_features[0] == 'None':
        global_features = []


    features = sorted(features)
    global_features = sorted(global_features)

    half_frame_length = False
    if frame_length == 90:
        half_frame_length = True


    print('Loading data...')
    main, left, right, global_feats, labels, ids = organize_data(features, global_features, use_ssp, half_frame_length)
    y = labels

    # make the global feats equal to the last element
    if len(global_features) > 0:
        global_feats = global_feats[:, :, -1]

        # repeat global feats num_global_feats_repeat times
        if args.num_global_feats_repeat > 0:
            global_feats = np.repeat(global_feats, args.num_global_feats_repeat, axis=1)

    print('Loaded data.')

    # normalization
    if not args.no_scaling:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_norm = min_max_scaler.fit_transform(main.reshape(-1, main.shape[1]*main.shape[2]))
        X_norm = X_norm.reshape(main.shape)

        min_max_scaler = preprocessing.MinMaxScaler()
        X2_norm = min_max_scaler.fit_transform(left.reshape(-1, left.shape[1]*left.shape[2]))
        X2_norm = X2_norm.reshape(left.shape)

        min_max_scaler = preprocessing.MinMaxScaler()
        X3_norm = min_max_scaler.fit_transform(right.reshape(-1, right.shape[1]*right.shape[2]))
        X3_norm = X3_norm.reshape(right.shape)
    else:
        X_norm = main
        X2_norm = left
        X3_norm = right

    X = np.stack((X_norm, X2_norm, X3_norm), axis=1)


    iterate = []

    # training split types: 70:30, 10-fold, loso-subject, loso-session
    if training_split == "70:30":
        iterate = [0]
    elif training_split == "10-fold":
        iterate = [0,1,2,3,4,5,6,7,8,9]
    elif training_split == "loso-subject":
        iterate = np.array(list(sorted(set(ids))))
    elif training_split == "loso-session":
        sessions = set()
        for i in range(len(ids)):
            sessions.add(ids[i][:2])
        iterate = np.array(sorted(list(sessions)))

    # only for 10-fold
    kfold = KFold(n_splits=10, shuffle=True)
    tenFoldSplits = list(kfold.split(X, y))

    aggregated_metrics = []

    for i in tqdm.tqdm(iterate):
        print("Training split is: ", training_split)
        print("Training fold / subject / session is: ", i)


        # shuffle and split
        # train_index, test_index = shuffle_train_test_split(len(y), 0.8)
        print("Making test-train splits")

        if training_split == "70:30":
            x_ids = list(range(len(X)))
            X_train_ids, X_test_ids, y_train, y_test = train_test_split(x_ids, y, test_size=0.3)

        elif training_split == "10-fold":
            # the value of iterate[i] is the fold number
            X_train_ids, X_test_ids = tenFoldSplits[i]
            y_train = y[X_train_ids]
            y_test = y[X_test_ids]

        elif training_split == "loso-subject":
            # the value of iterate[i] is the subject id for the test set!
            X_train_ids = []
            X_test_ids = []
            for j in range(len(y)):
                if ids[j] == i:
                    X_test_ids.append(j)
                else:
                    X_train_ids.append(j)
            y_train = y[X_train_ids]
            y_test = y[X_test_ids]

        elif training_split == "loso-session":
            # the value of iterate[i] is the session id for the test set!
            X_train_ids = []
            X_test_ids = []
            for j in range(len(y)):
                if ids[j][:2] == i:
                    X_test_ids.append(j)
                else:
                    X_train_ids.append(j)

            y_train = y[X_train_ids]
            y_test = y[X_test_ids]
        else:
            print("Error: training split not recognized")
            exit()



        print("Done making test-train splits")




        # split people up

        X1_train = X[X_train_ids, 0, :]
        X2_train = X[X_train_ids, 1, :]
        X3_train = X[X_train_ids, 2, :]
        global_train = global_feats[X_train_ids] # global_feats is len(y) x 2 (or 1)
        X1_test = X[X_test_ids, 0, :]
        X2_test = X[X_test_ids, 1, :]
        X3_test = X[X_test_ids, 2, :]
        global_test = global_feats[X_test_ids]

        # if use_ssp == 2: # no need to zero out since we are just ignoring that encoder
        #     # zero out the main person
        #     X1_train = np.zeros((X1_train.shape[0],0, 1))
        #     X1_test = np.zeros((X1_test.shape[0],0, 1))


        print('Checking Assertions')
        assert not np.any(np.isnan(X1_train))
        assert not np.any(np.isnan(X2_train))
        assert not np.any(np.isnan(X3_train))
        assert not np.any(np.isnan(X1_test))
        assert not np.any(np.isnan(X2_test))
        assert not np.any(np.isnan(X3_test))

        assert not np.any(np.isnan(y_train))
        assert not np.any(np.isnan(y_test))
        print("Assertions Valid")

        print("Training")
        print(X1_train.shape)
        print(X2_train.shape)
        print(X3_train.shape)
        print(y_train.shape)
        print(global_train.shape)
        print('Number of positive samples in training: ',np.sum(y_train))
        print('Number of negative samples in training: ',len(y_train)-np.sum(y_train))

        print("Test")
        print(X1_test.shape)
        print(X2_test.shape)
        print(X3_test.shape)
        print(global_test.shape)
        print(y_test.shape)
        print('Number of positive samples in test: ', np.sum(y_test))
        print('Number of negative samples in test: ', len(y_test) - np.sum(y_test))

        feats = '_'.join(sorted(features + global_features))
        feats = 'i_paznet_temp' + '_' + feats

        if args.filter_scale != 1:
            feats = feats + '_filter_scale' + str(args.filter_scale)
        # if args.no_scaling:
        #     feats += '_no_scaling'
        if half_frame_length:
            feats += '_15fps'

        split_string = training_split + '_' + str(i)
        group_name = training_split
        if use_ssp == 1:
            split_string += '_ssp'
            group_name += '_ssp'
        if use_ssp == 2:
            split_string += '_full_ssp'
            group_name += '_full_ssp'

        if args.num_global_feats_repeat > 0:
            split_string += '_rep' + str(args.num_global_feats_repeat)
            group_name += '_rep' + str(args.num_global_feats_repeat)

        paznet = PazNet(inputs=sorted(features+global_features), frame_length=frame_length, filter_scale=args.filter_scale, num_global_feats_repeat=args.num_global_feats_repeat, use_ssp=args.use_ssp)

        if not args.test:
            # create paznet
            paznet.create_model()

            import wandb
            from wandb.keras import WandbCallback

            config = {'training_split':training_split, 'test_split_value':i}
            config.update(vars(args))

            print("Creating wandb")
            run = wandb.init(project='social-dining', group=feats + '_' + group_name, config=config, name=feats+ '_' + split_string)
            print("Wandb Run: ", run)


            print("Creating lr scheduler")
            # learning rate decay
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=1000,
                decay_rate=decay_rate)

            # early stopping
            print('Creating model callbacks')
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

            # save the best model by measuring F1-score
            mc = ModelCheckpoint("checkpoints/" + feats + '_' + split_string + '_' + str(
                learning_rate) + "_" + str(decay_rate) + "_" + str(l2_value) + '_' + str(batch_size) + ".h5",
                                monitor='val_get_f1', mode='max', verbose=1, save_best_only=True)

            print('Compiling model')
            paznet.model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=binary_crossentropy, metrics=[get_f1, paznet.metrics])


            x_ids = list(range(len(X1_train)))

            X_train_ids, X_val_ids, y_train, y_val = train_test_split(x_ids, y_train, test_size=0.2)


            X1_val = X1_train[X_val_ids]
            X2_val = X2_train[X_val_ids]
            X3_val = X3_train[X_val_ids]
            global_val = global_feats[X_val_ids] # global_feats is len(y) x 2 (or 1)

            X1_train = X1_train[X_train_ids]
            X2_train = X2_train[X_train_ids]
            X3_train = X3_train[X_train_ids]
            global_train = global_feats[X_train_ids] # global_feats is len(y) x 2 (or 1)


            print('Starting training')
            # history = paznet.model.fit(x=[X1_train, X2_train, X3_train, global_train], y=y_train, epochs=epoch,
            #                     batch_size=batch_size, validation_data=([X1_test, X2_test, X3_test, global_test], y_test), callbacks=[es, mc, WandbCallback()])

            history = paznet.model.fit(x=[X2_train, X3_train, X1_train, global_train], y=y_train, epochs=epoch,
                                batch_size=batch_size, validation_data=([X2_val, X3_val, X1_val, global_val], y_val), callbacks=[es, mc, WandbCallback()])

            # history = paznet.model.fit(x=[X2_train, X3_train, X1_train, global_train], y=y_train, epochs=epoch,
            #                     batch_size=batch_size, validation_data=([X2_test, X3_test, X1_test, global_test], y_test), callbacks=[es, mc, WandbCallback()])

            # history = paznet.model.fit(x=[X2_train, X3_train, X1_train, global_train], y=y_train, epochs=epoch,
            #                     batch_size=batch_size, validation_split=.2, callbacks=[es, mc, WandbCallback()])



            model_path = "checkpoints/" + feats + '_' + split_string + '_' + str(
                learning_rate) + "_" + str(decay_rate) + "_" + str(l2_value) + '_' + str(batch_size) + ".h5"
            print("Loading model from: ", model_path)
            custom_objects = {"get_f1": get_f1,}
            with keras.utils.custom_object_scope(custom_objects):
                paznet.model = keras.models.load_model(model_path)

                predicted = paznet.model.predict(x=[X2_test, X3_test, X1_test, global_test], batch_size=batch_size)
                # predicted = np.argmax(predicted, axis=1)
                predicted = np.round(predicted)

                from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score
                print(predicted)
                print(y_test)

                # compute metrics
                f1 = f1_score(y_test, predicted)
                tp = precision_score(y_test, predicted, pos_label=1)
                fp = precision_score(y_test, predicted, pos_label=0)
                tn = precision_score(y_test, predicted,  pos_label=0)
                fn = precision_score(y_test, predicted, pos_label=1)
                acc = accuracy_score(y_test, predicted)
                prec = precision_score(y_test, predicted)
                recall = recall_score(y_test, predicted)
                auc = roc_auc_score(y_test, predicted)
                prc = average_precision_score(y_test, predicted)
                    
                evaluate_metrics = {'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'acc': acc, 'prec': prec, 'recall': recall, 'auc': auc, 'prc': prc}



                # log these metrics
                # update wandb summary
                # run.summary['best_loss'] = loss
                run.summary['best_f1'] = f1
                run.summary['best_tp'] = tp
                run.summary['best_fp'] = fp
                run.summary['best_tn'] = tn
                run.summary['best_fn'] = fn
                run.summary['best_acc'] = acc
                run.summary['best_prec'] = prec
                run.summary['best_recall'] = recall
                run.summary['best_auc'] = auc
                run.summary['best_prc'] = prc
                
            
            wandb.finish()
        else:
            # load model
            model_path = "checkpoints/" + feats + '_' + split_string + '_' + str(
                learning_rate) + "_" + str(decay_rate) + "_" + str(l2_value) + '_' + str(batch_size) + ".h5"
            print("Loading model from: ", model_path)
            custom_objects = {"get_f1": get_f1,}
            with keras.utils.custom_object_scope(custom_objects):
                paznet.model = keras.models.load_model(model_path)

            paznet.model.summary()


            predicted_train = paznet.model.predict(x=[X2_train, X3_train, X1_train, global_train], batch_size=batch_size)

            # import pdb; pdb.set_trace()


            tf.keras.utils.plot_model(paznet.model, to_file='model.png', show_shapes=True)

            predicted = paznet.model.predict(x=[X2_test, X3_test, X1_test, global_test], batch_size=batch_size)
            predicted = np.round(predicted)

            print(predicted)
            print(y_test)
            from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score, fbeta_score, matthews_corrcoef, jaccard_score

            # compute metrics
            f1 = f1_score(y_test, predicted)
            fbeta = fbeta_score(y_test, predicted, beta=2)
            mcc = matthews_corrcoef(y_test, predicted)
            jaccard = jaccard_score(y_test, predicted)


            tp = precision_score(y_test, predicted, pos_label=1)
            fp = precision_score(y_test, predicted, pos_label=0)
            tn = precision_score(y_test, predicted,  pos_label=0)
            fn = precision_score(y_test, predicted, pos_label=1)
            acc = accuracy_score(y_test, predicted)
            prec = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            auc = roc_auc_score(y_test, predicted)
            prc = average_precision_score(y_test, predicted)

            # f1 = f1_score(y_test, predicted, average='macro')
            # tp = precision_score(y_test, predicted, pos_label=1)
            # fp = precision_score(y_test, predicted, pos_label=0)
            # tn = precision_score(y_test, predicted,  pos_label=0)
            # fn = precision_score(y_test, predicted, pos_label=1)
            # acc = accuracy_score(y_test, predicted)
            # prec = precision_score(y_test, predicted, average='macro')
            # recall = recall_score(y_test, predicted, average='macro')
            # auc = roc_auc_score(y_test, predicted, average='macro')
            # prc = average_precision_score(y_test, predicted, average='macro')
            
            evaluate_metrics = {'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'acc': acc, 'prec': prec, 'recall': recall, 'auc': auc, 'prc': prc, 'fbeta':fbeta, 'jaccard':jaccard, 'mcc':mcc}



            aggregated_metrics.append(evaluate_metrics)
    # now present aggregated metrics

    print("Aggregated metrics: ", aggregated_metrics)
    print("Mean f1: {0:.4f} + {0:.4f}".format(np.mean([x['f1'] for x in aggregated_metrics]),    np.std([x['f1'] for x in aggregated_metrics]) ))
    print("Mean fbeta: {0:.4f} + {0:.4f}".format(np.mean([x['fbeta'] for x in aggregated_metrics]),    np.std([x['fbeta'] for x in aggregated_metrics]) ))
    print("Mean jaccard: {0:.4f} + {0:.4f}".format(np.mean([x['jaccard'] for x in aggregated_metrics]),    np.std([x['jaccard'] for x in aggregated_metrics]) ))
    print("Mean mcc: {0:.4f} + {0:.4f}".format(np.mean([x['mcc'] for x in aggregated_metrics]),    np.std([x['mcc'] for x in aggregated_metrics]) ))

    print("Mean acc: {0:.4f} + {0:.4f}".format(np.mean([x['acc'] for x in aggregated_metrics]),    np.std([x['acc'] for x in aggregated_metrics]) ))

    print("Mean prec: {0:.4f} + {0:.4f}".format(np.mean([x['prec'] for x in aggregated_metrics]),    np.std([x['prec'] for x in aggregated_metrics]) ))


    print("Mean recall: {0:.4f} + {0:.4f}".format(np.mean([x['recall'] for x in aggregated_metrics]),    np.std([x['recall'] for x in aggregated_metrics]) ))


    # print("Aggregated metrics: ", aggregated_metrics)
    # print("Mean f1: ", np.mean([x['f1'] for x in aggregated_metrics]))

    # print("Mean acc: ", np.mean([x['acc'] for x in aggregated_metrics]))
    # print("Mean prec: ", np.mean([x['prec'] for x in aggregated_metrics]))
    # print("Mean recall: ", np.mean([x['recall'] for x in aggregated_metrics]))
    # print("Mean auc: ", np.mean([x['auc'] for x in aggregated_metrics]))
    # print("Mean prc: ", np.mean([x['prc'] for x in aggregated_metrics]))


if __name__ == '__main__':
    # arg parse the features lists
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', nargs='+', type=str, default=['body', 'face', 'gaze', 'headpose', 'speaking'],
                        help='features to use')
    parser.add_argument('--global_features', nargs='+', type=str, default=['num_bites', 'time_since_last_bite']
                        , help='global features to use')
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_value', type=float, default=0.0001, help='l2 regularization value')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='learning rate decay rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--training_split', type=str, default='70:30', help='training split')
    parser.add_argument('--use_ssp', type=int, default=0, help='whether to use social signal processing-like method')
    parser.add_argument('--frame_length', type=int, default=180, help='frames to sample in a 6s sample')
    parser.add_argument('--filter_scale', type=int, default=1, help='how much to scale the number of filters in the conv net by')
    parser.add_argument('--no_scaling', type=int, default=1, help='whether to scale the data')
    parser.add_argument('--test', type=int, default=0, help='whether to test the model')
    parser.add_argument('--num_global_feats_repeat', type=int, default=1, help='number of times to repeat the global features')

    args = parser.parse_args()

    # train()
    train(args)
