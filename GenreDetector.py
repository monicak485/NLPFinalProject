# NLP Final Project
# Trigram Model to detect song genre based on lyrics
import csv

def loadData():
    with open('lyrics.csv', 'rt') as f:
        reader = csv.reader(f)
        songData = list(reader)
    labels = list()
    lyrics = list()
    for song in songData:
        labels.append(song[4])
        lyrics.append(song[5])

    for x in range(len(lyrics)):
        print(repr(lyrics[x]) + " " + repr(labels[x]))

def main():
    loadData()


main()
