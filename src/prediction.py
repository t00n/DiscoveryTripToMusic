import sys
from tqdm import tqdm
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression

from parser import *
from features import *

def get_features_vectors(filename):
    header = read_header_csv(filename)
    songs = []
    for index, row in tqdm(header.iterrows()):
        data = read_song_csv(int(row[0]))
        features = create_song_features(data)
        songs.append(features)
    songs = np.array(songs)
    return header, songs

def prediction(training_file, test_file, output_file):
    output, training_set = get_features_vectors(training_file)
    _, test_set = get_features_vectors(test_file)

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

    write_prediction_csv(output_file, composers, instruments, styles, years, tempos)

if __name__ == '__main__':
    prediction(*sys.argv[1:])