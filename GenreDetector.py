# NLP Final Project
# Trigram Model to detect song genre based on lyrics
import sklearn as sk
from sklearn import feature_extraction
from sklearn import naive_bayes
from sklearn import pipeline
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv
from random import randint
import nltk
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer


def loadData():
    with open('lyrics.csv', 'rt') as f:
        reader = csv.reader(f)
        songData = list(reader)
    labels = list()
    lyrics = list()

    for song in songData:
        labels.append(song[4])
        song[5].replace('\n', ' ')
        lyrics.append(song[5])

    #for x in range(len(lyrics)):
        #print(lyrics[x])

    return lyrics, labels

def tokenize(lyrics, labels):
    lyricBigrams = list()
    index = 0
    labelBigrams = list()
    tokenizer = RegexpTokenizer(r'\w+')
    for song in lyrics:
        token=tokenizer.tokenize(song)
        bigrams=ngrams(token,2)
        for bigram in bigrams:
            lyricBigrams.append(bigram)
            labelBigrams.append(labels[index])
        index += 1

    for x in range(len(lyricBigrams)):
        print(lyricBigrams[x])
        print(" is ")
        print(labelBigrams[x])



def main():
    lyrics, labels = loadData()
    lyricsShortened = list()
    labelsShortened = list()
    for x in range(100):
        lyricsShortened.append(lyrics[randint(1, len(lyrics))])
        labelsShortened.append(labels[randint(1, len(labels))])

    tokenize(lyricsShortened, labelsShortened)
    #makeClassifier(lyrics, labels)
    #makeClassifier(lyricsShortened, labelsShortened)


main()
