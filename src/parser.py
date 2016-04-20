from pandas import DataFrame, read_csv, Series
from sklearn.cluster import KMeans, DBSCAN
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np
import csv
from tqdm import tqdm
import json

DATA_REP = "../data/"
HEADER_FILE = DATA_REP + "dataset-balanced.csv"
SONG_REP = DATA_REP + "songs/"

def read_song(id):
    song_file = SONG_REP + str(id) + ".csv"
    with open(song_file) as csvfile:
        reader = csv.reader(csvfile)
        return [list(map(lambda x: x.strip(), row)) for row in reader]

def read_header(f):
    return read_csv(HEADER_FILE, header=0, sep=';')

def parse_key_signature(data):
    try:
        one, two = list(filter(lambda x: x[2] == "Key_signature", data))[0][-2:]
        key_signature = (int(one), (1 if two == '"major"' else 2))
    except:
        key_signature = None
    return key_signature

def create_song_features(data):
    """ missing 
            * chords based features
            * phrases based features
        "proportion of strong notes" was replaced by "note with highest velocity"
    """
    time_signature = map(int, list(filter(lambda x: x[2] == "Time_signature", data))[0][-4:])
    notes = DataFrame(list(map(lambda x: [int(x[0]), int(x[1]), x[2], int(x[3]), int(x[4]), int(x[5])], filter(lambda x: x[2][:4] == "Note", data))))
    tempo = int(list(filter(lambda x: x[2] == "Tempo", data))[0][3])
    total_time = notes[1].max()
    def tempo_to_bpm(tempo):
        return 60000000/tempo
    bpm = tempo_to_bpm(tempo)
    # pitch based features
    def pitch():
        return notes[4]
    def proportion_high():
        return len(pitch()[pitch() >= 72]) / len(pitch())
    def proportion_medium():
        return len(pitch()[54 <= pitch()][pitch() < 72]) / len(pitch())
    def proportion_bass():
        return len(pitch()[pitch() < 54]) / len(pitch())
    # duration based features
    def duration():
        durations = []
        on = {}
        for i, note in notes.iterrows():
            t = (note[3], note[4])
            if str(t) in on:
                beg = on[str(t)]
                durations.append(note[1] - beg)
                del on[str(t)]
            else:
                on[str(t)] = note[1]
        return Series(durations)
    duration = duration()
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
    def silence_proportion():
        return silence.sum() / total_time
    # velocity based features
    def velocity():
        return notes[5]
    def note_highest_velocity():
        return notes.groupby(4).max()[5].idxmax()
    # density based features
    def notes_on():
        return notes[notes[2].str[:7] == 'Note_on'][1]
    notes_on = notes_on()
    unique, density = np.unique(DBSCAN(400).fit_predict(notes_on.values.reshape(-1, 1)), return_counts=True)

    return [bpm, pitch().max(), pitch().min(), pitch().mean(), pitch().std(), proportion_high(), proportion_medium(), proportion_bass(), duration.max(), duration.min(), duration.mean(), duration.std(), velocity().max(), velocity().min(), velocity().mean(), velocity().std(), note_highest_velocity(), density.mean(), density.std(), silence_proportion(), silence.mean(), silence.std(), *time_signature]

output = read_header(HEADER_FILE)

try:
    songs = json.load(open("cache.json"))
    songs = np.array(songs)
except:
    songs = []
    for index, row in tqdm(output.iterrows()):
        data = read_song(row['id'])
        features = create_song_features(data)
        songs.append(list(map(float, features)))
    songs = np.array(songs)
    json.dump(songs.tolist(), open("cache.json", "w"))


clf = SVC()
clf.fit(songs, output['Performer'])
res = clf.predict(songs)

clf.fit(songs, output['Inst.'])
res = clf.predict(songs)

clf.fit(songs, output['Style'])
res = clf.predict(songs)

clf.fit(songs, output['Year'])
res = clf.predict(songs)



