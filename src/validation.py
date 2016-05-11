from prediction import predict_clf, predict_lin
from features import get_output, TARGETS, get_all
from collections import defaultdict

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
    return len(list(filter(lambda x: x[0] != x[1], zip(i, out))))/len(out)

def MAPE_lin(out, i):
    assert(len(i) == len(out))
    return sum(map(lambda x: abs((x[0] - x[1])/x[0]), zip(i, out)))/len(out)

def mean(errors):
    return sum(errors)/len(errors)

def cross_validation(target, type, features_on='all', error_clf=absolute_error_clf, error_lin=absolute_error_lin):
    errors = []
    for i in range(5):
        training_set, output, test_set = get_all(TRAINING_FILE % i, TEST_FILE % i, features_on)
        if type == 'cls':
            result = predict_clf(training_set, output[target], test_set)
            errors.append(error_clf(output[target], result))
        elif type == 'lin':
            result = predict_lin(training_set, output[target], test_set)
            errors.append(error_lin(output[target], result))
    return errors


if __name__ == '__main__':
    from settings import best_features
    errors = {}
    for target, t in TARGETS.items():
        print("Crossvalidation : ", target, t, best_features[target])
        v = cross_validation(target, t, best_features[target])
        errors[target] = v
    for target, error in errors.items():
        print(target, ' : ', error)

