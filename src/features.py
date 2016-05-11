from sklearn.cluster import KMeans, DBSCAN
from pandas import DataFrame, Series, merge
import numpy as np

from parser import *
from memoize import memoized

TARGETS = dict(zip(TARGETS_NAMES, ['cls', 'cls', 'cls', 'lin', 'lin']))

def get_features_vectors(filename, features_on='all'):
    header = read_output_csv(filename)
    songs = []
    for index, row in header.iterrows():
        features = create_song_features(int(row[0]), features_on)
        songs.append(features)
    songs = np.array(songs)
    return songs

@memoized
def get_output(f):
    header = read_header_csv(HEADER_FILE)
    output = read_output_csv(f)
    return merge(output, header, how='inner', left_on=[0], right_on=['id'])

def get_all(training_file, test_file, features_on):
    training_set = get_features_vectors(training_file, features_on)
    output = get_output(training_file)
    test_set = get_features_vectors(test_file, features_on)
    return training_set, output, test_set
    
def parse_key_signature(data):
    try:
        one, two = list(filter(lambda x: x[2] == "Key_signature", data))[0][-2:]
        key_signature = one + two
    except:
        key_signature = [0, ""]
    return key_signature

@memoized
def get_notes(id):
    data = read_song_csv(id)
    return DataFrame(list(filter(lambda x: x[2][:4] == "Note", data)))

@memoized
def get_differences(id):
    notes = get_notes(id)
    return notes[1].diff()

def features_proportion(id):
    notes = get_notes(id)
    pitch = notes[4]
    proportion_high = len(pitch[pitch >= 72]) / len(pitch)
    proportion_bass = len(pitch[pitch < 54]) / len(pitch)
    proportion_medium = 1 - proportion_high - proportion_bass
    return [proportion_high, proportion_medium, proportion_bass]

def features_pitch(id):
    notes = get_notes(id)
    pitch = notes[4]
    return [pitch.max(), pitch.min(), pitch.mean(), pitch.std()]

def features_total_time(id):
    notes = get_notes(id)
    return [notes[1].max()]

def features_duration(id):
    differences = get_differences(id)
    duration = differences.iloc[1::2]
    return [duration.max(), duration.min(), duration.mean(), duration.std()]

def features_silence(id):
    times = get_notes(id)[1]
    silence = get_differences(id).iloc[::2]
    silence_proportion = silence.sum() / times.max()
    return [silence_proportion, silence.mean(), silence.std()]

def features_highest_velocity(id):
    notes = get_notes(id)
    note_highest_velocity = notes.groupby(4).max()[5].idxmax()
    return [note_highest_velocity]

def features_velocity(id):
    velocity = get_notes(id)[5]
    return [velocity.max(), velocity.min(), velocity.mean(), velocity.std()]

def features_density(id):
    notes = get_notes(id)
    notes_on = notes[notes[2].str[:7] == 'Note_on'][1]
    unique, density = np.unique(DBSCAN(400).fit_predict(notes_on.values.reshape(-1, 1)), return_counts=True)
    return [density.max(), density.min(), density.mean(), density.std()]

def features_time_signature(id):
    data = read_song_csv(id)
    time_signature = list(filter(lambda x: x[2] == "Time_signature", data))[0][-4:]
    return [*time_signature]

features_list = [
    features_proportion,
    features_pitch,
    features_total_time,
    features_duration,
    features_silence,
    features_highest_velocity,
    features_velocity,
    features_density,
    features_time_signature
]

features_list = list(map(memoized, features_list))

NUMBER_OF_FEATURES = len(features_list)

def create_song_features(id, features_on = 'all'):
    """ missing 
        "proportion of strong notes" was replaced by "note with highest velocity"
    """
    if features_on == 'all':
        features_on = [True for i in range(NUMBER_OF_FEATURES)]
    features = []
    for i in range(NUMBER_OF_FEATURES):
        if features_on[i]:
            features.extend(features_list[i](id))
    return features
