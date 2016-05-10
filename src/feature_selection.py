from validation import *
from features import NUMBER_OF_FEATURES, TARGETS
from settings import save_best_features
from copy import copy
from random import randrange

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

def best_neighbour(target, type, features):
    best_features = copy(features)
    best_errors = mean(cross_validation(target, type, best_features))
    current_features = copy(features)
    for i in range(len(features)):
        current_features[i] = not current_features[i]
        if current_features != features and current_features != [False for i in range(NUMBER_OF_FEATURES)]:
            print("Trying neighbour ", current_features, "...")
            errors = mean(cross_validation(target, type, current_features))
            if errors < best_errors:
                best_features = copy(current_features)
                best_errors = errors
        current_features[i] = not current_features[i]
    return best_features, best_errors


def heuristic_feature_selection(target, type):
    print("Selecting best features for target %s..." % target)
    perm = permutations([True, False], NUMBER_OF_FEATURES)
    perm.remove([False for i in range(NUMBER_OF_FEATURES)])
    current_features = perm[randrange(len(perm))]
    improvement = True
    print('Initial : ', current_features)
    while improvement:
        improvement = False
        best_features, errors = best_neighbour(target, type, current_features)
        if best_features != current_features:
            print('Improvement : ', best_features)
            print('Error : ', errors)
            improvement = True
            current_features = best_features
        current_errors = errors
    print("Best configuration : ", current_features)
    print("Error : ", current_errors)
    return current_features

def feature_selection(target, type):
    print("Selecting best features for target %s..." % target)
    perm = permutations([True, False], NUMBER_OF_FEATURES)
    perm.remove([False for i in range(NUMBER_OF_FEATURES)])
    current_features = perm[0]
    current_errors = mean(cross_validation(target, type, current_features))
    for features_on in perm:
        print("Trying ", features_on, "...")
        errors = mean(cross_validation(target, type, features_on))
        if errors < current_errors:
            current_features = features_on
            current_errors = errors
            print('Improvement : ', current_features)
            print('Error : ', current_errors)
    print("Best configuration : ", current_features)
    print("Error : ", current_errors)
    return current_features


if __name__ == '__main__':
    best_features = {}
    for target, type in TARGETS.items():
        best_features[target] = feature_selection(target, type)
    save_best_features(best_features)
