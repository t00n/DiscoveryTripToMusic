from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import sys

from parser import write_prediction_csv
from features import *

def prediction(training_file, test_file, output_file):
    training_set = get_features_vectors(training_file)
    output = get_output(training_file)
    test_set = get_features_vectors(test_file)

    clf = SVC()
    linear = LinearRegression()

    clf.fit(training_set, output['Performer'])
    composers = clf.predict(test_set)

    clf.fit(training_set, output['Inst.'])
    instruments = clf.predict(test_set)

    clf.fit(training_set, output['Style'])
    styles = clf.predict(test_set)

    linear.fit(training_set, output['Year'])
    years = linear.predict(test_set)

    linear.fit(training_set, output['Tempo'])
    tempos = linear.predict(test_set)

    if output_file != "":
        write_prediction_csv(output_file, composers, instruments, styles, years, tempos)
    return composers, instruments, styles, years, tempos

if __name__ == '__main__':
    prediction(*sys.argv[1:])