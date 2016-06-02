import csv
from pandas import read_csv

from memoize import memoized

DATA_REP = "../data/"
HEADER_FILE = DATA_REP + "dataset-balanced.csv"
SONG_REP = DATA_REP + "songs/"
MFCC_REP = DATA_REP + "mfccs/"

TARGETS_NAMES = ['Performer', 'Inst.', 'Style', 'Year', 'Tempo']

""" IO """
@memoized
def read_song_csv(id):
    def parseElem(elem):
        elem = elem.strip()
        try:
            elem = float(elem)
        except:
            pass
        return elem
    song_file = SONG_REP + str(id) + ".csv"
    with open(song_file) as csvfile:
        reader = csv.reader(csvfile)
        return [[parseElem(col) for col in row] for row in reader]

@memoized
def read_mfcc_csv(id):
    res=[]
    mfcc_file = MFCC_REP + str(id) + ".csv"
    with open(mfcc_file) as csvfile:
        temp=float(csvfile.readline().split('[')[1].split(']')[0])
        res.append(temp)
        temps=csvfile.readline().split('\n')[0].split(',')
        for item in temps:
            res.append(float(item))
    return res
@memoized
def read_header_csv(f):
    return read_csv(f, header=0, sep=';')

@memoized
def read_output_csv(f):
    return read_csv(f, header=-1, sep=';')

def write_prediction_csv(filename, composers, instruments, styles, years, tempos):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(composers)):
            writer.writerow([composers[i], instruments[i], styles[i], years[i], tempos[i]])
