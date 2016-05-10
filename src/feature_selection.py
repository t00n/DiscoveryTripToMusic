from validation import *
from features import NUMBER_OF_FEATURES, TARGETS
from settings import save_best_features
from copy import copy

def permutations(ls, n):
    def permutations_rec(current, n):
        if n > 1:
            ret = []
            for elem1 in current:
                for elem2 in ls:
                    newelem = copy(elem1) + [elem2]
                    ret.append(newelem)
            return permutations_rec(ret, n - 1)
        else:
            return current
    return permutations_rec(list(map(lambda x: [x], ls)), n)

def feature_selection(target, type):
    print("Selecting best features for target %s..." % target)
    current_features = None
    current_errors = float('inf')
    perm = permutations([True, False], NUMBER_OF_FEATURES)
    perm.remove([False for i in range(NUMBER_OF_FEATURES)])
    for features_on in perm:
        print("Trying ", features_on, "...")
        errors = MAPE_all(cross_validation(target, type, features_on))
        if errors < current_errors:
            current_errors = errors
            current_features = features_on
        print("MAPE : ", errors)
    print("Best configuration : ", current_features)
    print("MAPE : ", current_errors)
    return current_features

if __name__ == '__main__':
    best_features = {}
    for target, type in TARGETS.items():
        best_features[target] = feature_selection(target, type)
    save_best_features(best_features)
