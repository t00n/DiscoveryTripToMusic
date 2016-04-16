from pandas import read_csv

data_repertory = "../data/"
header_file = data_repertory + "dataset-balanced.csv"
song_repertory = data_repertory + "songs/"


print(read_csv(header_file))
