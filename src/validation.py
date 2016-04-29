from prediction import *
from parser import read_output_csv
import pandas

TEST_FILE="test-data-file-%d.csv"
TRAINING_FILE="training-data-file-%d.csv"

def error_cls(i, out):
    assert(len(i) == len(out))
    return len(list(filter(lambda x: x[0] != x[1], zip(i, out))))

def error_lin(i, out):
    assert(len(i) == len(out))

def cross_validation():
    errors = [0 for i in range(5)]
    for i in range(5):
        composers, instruments, styles, years, tempos = prediction(TRAINING_FILE % i, TEST_FILE %i, "")
        output = read_output_csv(TEST_FILE % i)
        errors[0] = error_cls(list(output['Performer']), composers)
        errors[1] = error_cls(list(output['Inst.']), instruments)
        errors[2] = error_cls(list(output['Style']), styles)



if __name__ == '__main__':
    cross_validation()

