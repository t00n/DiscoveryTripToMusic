from pandas import DataFrame, read_csv, Series
import csv

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

def create_song_features(data):
    """ missing chords based features """
    notes = DataFrame(list(map(lambda x: [int(x[0]), int(x[1]), x[2], int(x[3]), int(x[4]), int(x[5])], filter(lambda x: x[2][:4] == "Note", data))))
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
    return [pitch().max(), pitch().min(), pitch().mean(), pitch().std(), proportion_high(), proportion_medium(), proportion_bass(), duration.max(), duration.min(), duration.mean(), duration.std()]


output = read_header(HEADER_FILE)
songs = []

for index, row in output.iterrows():
    data = read_song(row['id'])
    features = create_song_features(data)
    print(features)
    break

