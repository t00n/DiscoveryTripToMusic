from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
import sys


from parser import write_prediction_csv, TARGETS_NAMES
from features import *

def predict_clf(training_set, output, test_set):
    clf = OneVsOneClassifier(SVC(kernel='linear'))
    from sklearn import preprocessing
    training_set=preprocessing.scale(training_set)
    test_set=preprocessing.scale(test_set)

    clf.fit(training_set, output)
    return clf.predict(test_set)

def predict_lin(training_set, output, test_set):
    lin = LinearRegression()
    lin.fit(training_set, output)
    return lin.predict(test_set)

def prediction(training_file, test_file, features_on, output_file=''):
    clf = OneVsOneClassifier(SVC(kernel='linear'))
    lin = LinearRegression()

    training_set, output, test_file = get_all(training_file, test_file, features_on)
    results = [(lambda f: f(training_set, output[x], test_set))(predict_clf if TARGETS[x] == 'cls' else predict_lin) for x in TARGETS_NAMES]

    if output_file != "":
        write_prediction_csv(output_file, *results)
    return results

if __name__ == '__main__':
    from settings import best_features
    prediction(sys.argv[1], sys.argv[2], best_features, sys.argv[3])
