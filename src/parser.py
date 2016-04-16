from pandas import read_csv

data_repertory = "../data/"
header_file = data_repertory + "dataset-balanced.csv"
song_repertory = data_repertory + "songs/"


output = read_csv(header_file, header=0, sep=';')
songs = []

for index, row in output.iterrows():
    song_file = song_repertory + str(row['id']) + ".csv"
    # parse song file here
