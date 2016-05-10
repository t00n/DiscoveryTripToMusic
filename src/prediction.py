from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import sys

from parser import write_prediction_csv, TARGETS_NAMES
from features import *

def predict_clf(training_file, test_file, target, features_on):
    training_set, output, test_set = get_all(training_file, test_file, features_on)
    clf = SVC()
    clf.fit(training_set, output[target])
    return clf.predict(test_set)

def predict_lin(training_file, test_file, target, features_on):
    training_set, output, test_set = get_all(training_file, test_file, features_on)
    lin = LinearRegression()
    lin.fit(training_set, output[target])
    return lin.predict(test_set)

def get_all(training_file, test_file, features_on):
    training_set = get_features_vectors(training_file, features_on)
    output = get_output(training_file)
    test_set = get_features_vectors(test_file, features_on)
    return training_set, output, test_set

def prediction(training_file, test_file, features_on, output_file=''):
    clf = SVC()
    lin = LinearRegression()

    results = [(lambda f: f(training_file, test_file, x, features_on[x]))(predict_clf if TARGETS[x] == 'cls' else predict_lin) for x in TARGETS_NAMES]

    if output_file != "":
        write_prediction_csv(output_file, *results)
    return results

if __name__ == '__main__':
    from settings import best_features
    prediction(sys.argv[1], sys.argv[2], best_features, sys.argv[3])