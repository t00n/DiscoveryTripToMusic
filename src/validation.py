from prediction import prediction
from features import get_output

TEST_FILE="test-data-file-%d.csv"
TRAINING_FILE="training-data-file-%d.csv"

def error_cls(i, out):
    assert(len(i) == len(out))
    return len(list(filter(lambda x: x[0] != x[1], zip(i, out))))

def error_lin(i, out):
    assert(len(i) == len(out))
    return sum(map(lambda x: abs(x[0] - x[1]), zip(i, out)))

def cross_validation():
    errors = [[] for i in range(5)]
    for i in range(5):
        output = get_output(TEST_FILE % i)
        composers, instruments, styles, years, tempos = prediction(TRAINING_FILE % i, TEST_FILE %i, "")
        errors[0].append(error_cls(list(output['Performer']), composers))
        errors[1].append(error_cls(list(output['Inst.']), instruments))
        errors[2].append(error_cls(list(output['Style']), styles))
        errors[3].append(error_lin(list(output['Year']), years))
        errors[4].append(error_lin(list(output['Tempo']), tempos))
    return errors

if __name__ == '__main__':
    v = cross_validation()
    print(v)

