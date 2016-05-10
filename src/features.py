from sklearn.cluster import KMeans, DBSCAN
from pandas import DataFrame, Series, merge
import numpy as np

from parser import *
from memoize import memoized

NUMBER_OF_FEATURES = 9
TARGETS = dict(zip(TARGETS_NAMES, ['cls', 'cls', 'cls', 'lin', 'lin']))

@memoized
def get_features_vectors(filename, features_on='all'):
    header = read_output_csv(filename)
    songs = []
    for index, row in header.iterrows():
        data = read_song_csv(int(row[0]))
        features = create_song_features(data, features_on)
        songs.append(features)
    songs = np.array(songs)
    return songs

@memoized
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

def create_song_features(data, features_on = 'all'):
    """ missing 
        "proportion of strong notes" was replaced by "note with highest velocity"
    """
    if features_on == 'all':
        features_on = [True for i in range(NUMBER_OF_FEATURES)]
    features = []
    notes = DataFrame(list(filter(lambda x: x[2][:4] == "Note", data)))
    # pitch based features
    pitch = notes[4]
    if features_on[0]:
        proportion_high = len(pitch[pitch >= 72]) / len(pitch)
        proportion_bass = len(pitch[pitch < 54]) / len(pitch)
        proportion_medium = 1 - proportion_high - proportion_bass
        features.extend([proportion_high, proportion_medium, proportion_bass])
    if features_on[1]:
        features.extend([pitch.max(), pitch.min(), pitch.mean(), pitch.std()])
    # duration based features
    times = notes[1]
    total_time = times.max()
    differences = times.diff()
    if features_on[2]:
        features.append(total_time)
    if features_on[3]:
        duration = differences.iloc[1::2]
        features.extend([duration.max(), duration.min(), duration.mean(), duration.std()])
    if features_on[4]:
        silence = differences.iloc[::2]
        silence_proportion = silence.sum() / total_time
        features.extend([silence_proportion, silence.mean(), silence.std()])
    # velocity based features
    if features_on[5]:
        note_highest_velocity = notes.groupby(4).max()[5].idxmax()
        features.append(note_highest_velocity)
    if features_on[6]:
        velocity = notes[5]
        features.extend([velocity.max(), velocity.min(), velocity.mean(), velocity.std()])
    # density based features
    if features_on[7]:
        notes_on = notes[notes[2].str[:7] == 'Note_on'][1]
        unique, density = np.unique(DBSCAN(400).fit_predict(notes_on.values.reshape(-1, 1)), return_counts=True)
        features.extend([density.max(), density.min(), density.mean(), density.std()])
    # time signature
    if features_on[8]:
        time_signature = list(filter(lambda x: x[2] == "Time_signature", data))[0][-4:]
        features.extend([*time_signature])
    # chords based features
    # if features_on[5]:
    #     row = len(songs.readlines())
    #     times = notes[1]
    #     pitch = notes[4]
    #     for i in range (0,row):
    #         if times.iloc['i',:] == times.iloc['i+1',:] :
    #             pitch.iloc['i',:] = pitch.iloc['i',:] + pitch.iloc['i+1',:]
    #             chords = pitch.iloc['i',:]
    #             i = 1
    #             i +=1
    #             i
    #         else:
    #             chords = pitch.iloc['i',:]
    #             i = 1
    #             i +=1
    #             i
    # # phrases based features
    # if features_on[6]:
    #     row = len(songs.readlines())
    #     times = notes[1]
    #     pitch = notes[4]
    #     for i in range (0,row):
    #         if pitch.iloc['i',:] == pitch.iloc['i+1',:] :
    #             times.iloc['i',:] = times.iloc['i',:] + times.iloc['i+1',:]
    #             phrases = times.iloc['i',:]
    #             i = 0
    #             i +=1
    #             i
    #         else:
    #             phrases = times.iloc['i',:]
    #             i = 0
    #             i +=1
    #             i
    return features
