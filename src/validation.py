from prediction import prediction
from features import get_output

TEST_FILE="test-data-file-%d.csv"
TRAINING_FILE="training-data-file-%d.csv"

def absolute_error_clf(i, out):
    assert(len(i) == len(out))
    return len(list(filter(lambda x: x[0] != x[1], zip(i, out))))

def absolute_error_lin(i, out):
    assert(len(i) == len(out))
    return sum(map(lambda x: abs(x[0] - x[1]), zip(i, out)))

def MAPE_clf(out, i):
    assert(len(i) == len(out))
    return len(list(filter(lambda x: x[0] != x[1], zip(i, out))))/36

def MAPE_lin(out, i):
    assert(len(i) == len(out))
    return sum(map(lambda x: abs((x[0] - x[1])/x[0]), zip(i, out)))/36

def cross_validation(error_clf, error_lin):
    errors = [[] for i in range(5)]
    for i in range(5):
        output = get_output(TEST_FILE % i)
        composers, instruments, styles, years, tempos = prediction(TRAINING_FILE % i, TEST_FILE %i, "")
        errors[0].append(error_clf(list(output['Performer']), composers))
        errors[1].append(error_clf(list(output['Inst.']), instruments))
        errors[2].append(error_clf(list(output['Style']), styles))
        errors[3].append(error_lin(list(output['Year']), years))
        errors[4].append(error_lin(list(output['Tempo']), tempos))
    return errors

def total_error(errors):
    return sum([sum(x) for x in errors])/25

if __name__ == '__main__':
    v = cross_validation(MAPE_clf, MAPE_lin)
    errors = total_error(v)
    print(v)
    print(errors)

