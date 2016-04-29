from sklearn.cluster import KMeans, DBSCAN
from pandas import DataFrame, Series
import numpy as np


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
    key_signature = parse_key_signature(data)
    notes = DataFrame(list(filter(lambda x: x[2][:4] == "Note", data)))
    total_time = notes[1].max()
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

    return [pitch().max(), pitch().min(), pitch().mean(), pitch().std(), proportion_high(), proportion_medium(), proportion_bass(), duration.max(), duration.min(), duration.mean(), duration.std(), velocity().max(), velocity().min(), velocity().mean(), velocity().std(), note_highest_velocity(), density.mean(), density.std(), silence_proportion(), silence.mean(), silence.std(), *time_signature, *key_signature]
