from sklearn.cluster import KMeans, DBSCAN
from pandas import DataFrame, Series, merge
import numpy as np
from tqdm import tqdm

from parser import *

def get_features_vectors(filename):
    header = read_output_csv(filename)
    songs = []
    for index, row in tqdm(header.iterrows()):
        data = read_song_csv(int(row[0]))
        features = create_song_features(data)
        songs.append(features)
    songs = np.array(songs)
    return songs

def get_output(f):
    header = read_header_csv(HEADER_FILE)
    output = read_output_csv(f)
    return merge(output, header, how='inner', left_on=[0], right_on=['id'])

def parse_key_signature(data):
    try:
        one, two = list(filter(lambda x: x[2] == "Key_signature", data))[0][-2:]
        key_signature = one + two
    except:
        key_signature = [0, ""]
    return key_signature

def create_song_features(data):
    """ missing 
            * chords based features
            * phrases based features
        "proportion of strong notes" was replaced by "note with highest velocity"
    """
    time_signature = list(filter(lambda x: x[2] == "Time_signature", data))[0][-4:]
    notes = DataFrame(list(filter(lambda x: x[2][:4] == "Note", data)))
    total_time = notes[1].max()
    notes_on = notes[notes[2].str[:7] == 'Note_on'][1]
    # pitch based features
    pitch = notes[4]
    proportion_high = len(pitch[pitch >= 72]) / len(pitch)
    proportion_bass = len(pitch[pitch < 54]) / len(pitch)
    proportion_medium = 1 - proportion_high - proportion_bass
    # duration based features
    duration = notes_on.diff()
    def silence():
        silences = []
        on = None
        for i, note in notes.iterrows():
            if on:
                silences.append(note[1] - on)
                on = None
            else:
                on = note[1]
        return Series(silences)
    silence = silence()
    silence_proportion = silence.sum() / total_time
    # velocity based features
    velocity = notes[5]
    note_highest_velocity = notes.groupby(4).max()[5].idxmax()
    # density based features
    unique, density = np.unique(DBSCAN(400).fit_predict(notes_on.values.reshape(-1, 1)), return_counts=True)

    return [pitch.max(), pitch.min(), pitch.mean(), pitch.std(), proportion_high, proportion_medium, proportion_bass, duration.max(), duration.min(), duration.mean(), duration.std(), velocity.max(), velocity.min(), velocity.mean(), velocity.std(), note_highest_velocity, density.mean(), density.std(), silence_proportion, silence.mean(), silence.std(), *time_signature]

def number_of_features():
    v = [[0, 0, "Time_signature", 0, 0, 0, 0], [0, 0, "Note_on", 0, 0, 0, 0]]
    print(v)
    return len(create_song_features(v))