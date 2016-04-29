import csv
from pandas import read_csv

DATA_REP = "../data/"
HEADER_FILE = DATA_REP + "dataset-balanced.csv"
SONG_REP = DATA_REP + "songs/"

""" IO """
def read_song_csv(id):
    def parseElem(elem):
        elem = elem.strip()
        try:
            elem = int(elem)
        except:
            pass
        return elem
    song_file = SONG_REP + str(id) + ".csv"
    with open(song_file) as csvfile:
        reader = csv.reader(csvfile)
        return [[parseElem(col) for col in row] for row in reader]

def read_header_csv(f):
    return read_csv(f, header=0, sep=';')

def read_output_csv(f):
    return read_csv(f, header=-1, sep=';')

def write_prediction_csv(filename, composers, instruments, styles, years, tempos):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(composers)):
            writer.writerow([composers[i], instruments[i], styles[i], years[i], tempos[i]])
