import numpy as np
import pandas as pd
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
from preprocessing import load_data
from preprocessing import load_data_single
from sklearn.metrics import classification_report
import pickle


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


model_list = [SVM(c=1, kernel='linear'), svm.SVC(C=1, gamma='scale', kernel='rbf'), SGD(), MLP()]
model_name = ["Linear SVM", "RBF SVM", "SGD", "MLP"]

X_train, X_test, y_train, y_test = load_data_single(split=True)
model = model_list[0]
model.fit(X_train, y_train.ravel())
with open('linear_svm_single.pkl','wb') as f:
    pickle.dump(model,f)
print("single svm saved")
'''
X_train, X_test, y_train, y_test = load_data(split=True)
model = model_list[0]
model.fit(X_train, y_train.ravel())
with open('linear_svm_social.pkl','wb') as f:
    pickle.dump(model,f)
print("social svm saved")
'''


i=0

for model in model_list:
    print(model_name[i])
    i = i+1

    print("social domain:")
    X, y = load_data(split=False)
    print("10 fold starting...")

    # 10 folds validation
    skf = StratifiedKFold(n_splits=10)
    total_accuracy = []
    total_precision = []
    total_recall = []
    total_f1score = []

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        model.fit(X_train, y_train.ravel())

        prediction = model.predict(X_test)

        report = classification_report(y_test, prediction, output_dict=True)
        accuracy = model.score(X_test, y_test)
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']
        print(accuracy)
        print(macro_precision)
        print(macro_recall)
        print(macro_f1)

        total_accuracy.append(accuracy)
        total_precision.append(macro_precision)
        total_recall.append(macro_recall)
        total_f1score.append(macro_f1)

    print("accuracy: ", np.mean(total_accuracy))
    print("precision: ", np.mean(total_precision))
    print("recall: ", np.mean(total_recall))
    print("f1 score: ", np.mean(total_f1score))


