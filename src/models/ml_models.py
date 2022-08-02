import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
from sklearn.metrics import classification_report
import pickle

import argparse
from data_utils import organize_data
from sklearn.model_selection import KFold
import random
import tqdm

def SVM(c, kernel):
    clf = svm.SVC(C=c, kernel=kernel)
    return clf


def SGD():
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='log')
    return clf


def MLP():
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(random_state=1, max_iter=300)
    return clf


model_dict = {'Linear SVM':SVM(c=1, kernel='linear'), 'RBF SVM':svm.SVC(C=1, gamma='scale', kernel='rbf'), 'SGD':SGD(), 'MLP':MLP()}


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

    print('Loading data...')
    main, left, right, global_feats, labels, ids = organize_data(features, global_features, use_ssp)
    y = labels
    print('Loaded data.')
    global_feats = global_feats[:, :, -1]

    # repeat global feats num_global_feats_repeat times
    if args.num_global_feats_repeat > 0:
        global_feats = np.repeat(global_feats, args.num_global_feats_repeat, axis=1)

    # normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    X_norm = min_max_scaler.fit_transform(main.reshape(-1, main.shape[1]*main.shape[2]))
    X_norm = X_norm.reshape(main.shape)

    min_max_scaler = preprocessing.MinMaxScaler()
    X2_norm = min_max_scaler.fit_transform(left.reshape(-1, left.shape[1]*left.shape[2]))
    X2_norm = X2_norm.reshape(left.shape)

    min_max_scaler = preprocessing.MinMaxScaler()
    X3_norm = min_max_scaler.fit_transform(right.reshape(-1, right.shape[1]*right.shape[2]))
    X3_norm = X3_norm.reshape(right.shape)

    X = np.stack((X_norm, X2_norm, X3_norm), axis=1)


    iterate = []

    aggregated_metrics = []

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



        # print("Done making test-train splits")


        # split people up

        X1_train = X[X_train_ids, 0, :]
        X2_train = X[X_train_ids, 1, :]
        X3_train = X[X_train_ids, 2, :]
        global_train = global_feats[X_train_ids] # global_feats is len(y) x 2 (or 1)

        X1_test = X[X_test_ids, 0, :]
        X2_test = X[X_test_ids, 1, :]
        X3_test = X[X_test_ids, 2, :]
        global_test = global_feats[X_test_ids]


        # print('Checking Assertions')
        assert not np.any(np.isnan(X1_train))
        assert not np.any(np.isnan(X2_train))
        assert not np.any(np.isnan(X3_train))
        assert not np.any(np.isnan(X1_test))
        assert not np.any(np.isnan(X2_test))
        assert not np.any(np.isnan(X3_test))

        assert not np.any(np.isnan(y_train))
        assert not np.any(np.isnan(y_test))
        # print("Assertions Valid")

        # print("Training")
        # print(X1_train.shape)
        # print(X2_train.shape)
        # print(X3_train.shape)
        # print(y_train.shape)
        print('Number of positive samples in training: ',np.sum(y_train))
        print('Number of negative samples in training: ',len(y_train)-np.sum(y_train))

        # print("Test")
        # print(X1_test.shape)
        # print(X2_test.shape)
        # print(X3_test.shape)
        # print(y_test.shape)
        print('Number of positive samples in test: ', np.sum(y_test))
        print('Number of negative samples in test: ', len(y_test) - np.sum(y_test))

        # flatten all samples to 1d
        X1_train = X1_train.reshape(X1_train.shape[0], -1)
        X2_train = X2_train.reshape(X2_train.shape[0], -1)
        X3_train = X3_train.reshape(X3_train.shape[0], -1)
        X1_test = X1_test.reshape(X1_test.shape[0], -1)
        X2_test = X2_test.reshape(X2_test.shape[0], -1)
        X3_test = X3_test.reshape(X3_test.shape[0], -1)

        print(X1_train.shape)
        print(global_train.shape)

        # now put it all together
        X_train = np.concatenate((X1_train, X2_train, X3_train, global_train), axis=1)
        X_test = np.concatenate((X1_test, X2_test, X3_test, global_test), axis=1)

        # X_train = np.stack((X1_train, X2_train, X3_train), axis=1)
        # X_test = np.stack((X1_test, X2_test, X3_test), axis=1)

        # print(X_train.shape)

        if args.model == 'random':
            continue
        elif args.model == 'ones':
            prediction = np.ones_like(y_test)
        elif args.model == 'zeros':
            prediction = np.zeros_like(y_test)

        else:
            # create paznet
            model = model_dict[args.model]

            print('Training model')        
            model.fit(X_train, y_train.ravel())

            prediction = model.predict(X_test)

        print(prediction.shape)
        predicted = prediction
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score, fbeta_score, matthews_corrcoef, jaccard_score

        f1 = f1_score(y_test, predicted)
        acc = accuracy_score(y_test, predicted)
        prec = precision_score(y_test, predicted)
        recall = recall_score(y_test, predicted)
        fbeta = fbeta_score(y_test, predicted, beta=2)
        mcc = matthews_corrcoef(y_test, predicted)
        jaccard = jaccard_score(y_test, predicted)


        # report = classification_report(y_test, prediction, output_dict=True)
        # # accuracy = model.score(X_test, y_test)
        # accuracy = report['accuracy']
        # macro_precision = report['macro avg']['precision']
        # macro_recall = report['macro avg']['recall']
        # macro_f1 = report['macro avg']['f1-score']
        # print('Accuracy', accuracy)
        # print('Precision', macro_precision)
        # print('Recall', macro_recall)
        # print('F1', macro_f1)
        # print(report)
        evaluate_metrics = {'acc':acc, 'precision':prec, 'recall':recall, 'f1':f1, 'fbeta':fbeta, 'jaccard':jaccard, 'mcc':mcc}
        # evaluate_metrics = {'acc':accuracy, 'precision':macro_precision, 'recall':macro_recall, 'f1':macro_f1}

        # evaluate_metrics = {'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'acc': acc, 'prec': prec, 'recall': recall, 'auc': auc, 'prc': prc}

        aggregated_metrics.append(evaluate_metrics)
        print("Mean acc: ", np.mean([x['acc'] for x in aggregated_metrics]))
        print("Mean precision: ", np.mean([x['precision'] for x in aggregated_metrics]))
        print("Mean recall: ", np.mean([x['recall'] for x in aggregated_metrics]))
        print("Mean f1: ", np.mean([x['f1'] for x in aggregated_metrics]))

        print("Mean f1: {0:.4f} + {0:.4f}".format(np.mean([x['f1'] for x in aggregated_metrics]),    np.std([x['f1'] for x in aggregated_metrics]) ))
        print("Mean fbeta: {0:.4f} + {0:.4f}".format(np.mean([x['fbeta'] for x in aggregated_metrics]),    np.std([x['fbeta'] for x in aggregated_metrics]) ))
        print("Mean jaccard: {0:.4f} + {0:.4f}".format(np.mean([x['jaccard'] for x in aggregated_metrics]),    np.std([x['jaccard'] for x in aggregated_metrics]) ))
        print("Mean mcc: {0:.4f} + {0:.4f}".format(np.mean([x['mcc'] for x in aggregated_metrics]),    np.std([x['mcc'] for x in aggregated_metrics]) ))



if __name__ == '__main__':
    # arg parse the features lists
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Linear SVM', help='model to use')
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
    parser.add_argument('--num_global_feats_repeat', type=int, default=0, help='number of times to repeat the global features')

    args = parser.parse_args()

    # train()
    train(args)