from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
import sys

from features import *

def prediction(training_file, test_file, output_file):
    training_set = get_features_vectors(training_file)
    output = get_output(training_file)
    test_set = get_features_vectors(test_file)

    clf = LinearSVC()
    linear = LinearRegression()

    clf.fit(training_set, output[1])
    composers = clf.predict(test_set)

    clf.fit(training_set, output[3])
    instruments = clf.predict(test_set)

    clf.fit(training_set, output[4])
    styles = clf.predict(test_set)

    linear.fit(training_set, output[5])
    years = linear.predict(test_set)

    linear.fit(training_set, output[6])
    tempos = linear.predict(test_set)

    if output_file != "":
        write_prediction_csv(output_file, composers, instruments, styles, years, tempos)
    return composers, instruments, styles, years, tempos

if __name__ == '__main__':
    prediction(*sys.argv[1:])