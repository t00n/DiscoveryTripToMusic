from prediction import predict_clf, predict_lin
from features import get_output, TARGETS

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

def MAPE_all(errors):
    return sum(errors)/5

def cross_validation(target, type, features_on='all', error_clf=MAPE_clf, error_lin=MAPE_lin):
    errors = []
    for i in range(5):
        output = get_output(TEST_FILE % i)
        if type == 'cls':
            result = predict_clf(TRAINING_FILE % i, TEST_FILE % i, target, features_on)
            errors.append(error_clf(output[target], result))
        elif type == 'lin':
            result = predict_lin(TRAINING_FILE % i, TEST_FILE % i, target, features_on)
            errors.append(error_lin(output[target], result))
    return errors


if __name__ == '__main__':
    from settings import best_features
    total = 0
    for target, t in TARGETS.items():
        print("Crossvalidation : ", target, t, best_features[target])
        v = cross_validation(target, t, best_features[target])
        errors = MAPE_all(v)
        print('MAPE : %f' % errors)
        total += errors
    print('Total MAPE : %f' % (total / len(TARGETS)))

