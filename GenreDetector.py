# NLP Final Project
# Trigram Model to detect song genre based on lyrics
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import csv
from random import randint
import nltk
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


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

    return lyrics, labels


def makeClassifier(data, labels):
    # create CV
    cv = sk.feature_extraction.text.CountVectorizer()
    # get splits based on kfxv
    kf = sk.model_selection.KFold(n_splits=10)
    trainData = []
    trainLabels = []
    testData = []
    testLabels = []
    avgAccuracy = 0
    avgf1 = 0
    for train_index, test_index in kf.split(data, labels):
        #sort train and test arrays based on kf splits
        for x in np.nditer(train_index):
            trainData.append(data[x])
            trainLabels.append(labels[x])
        for x in np.nditer(test_index):
            testData.append(data[x])
            testLabels.append(labels[x])
        # create the CV arrays based on train and test data
        train_cv = cv.fit_transform(trainData)
        test_cv = cv.transform(testData)
        # normalize
        tfidf = TfidfTransformer()
        tfidf_features = tfidf.fit_transform(train_cv)
        # classify this fold
        accuracy = classifyModel(tfidf_features, test_cv, trainLabels, testLabels)
        avgAccuracy += accuracy
    # get avg scores over all folds
    accuracy = avgAccuracy/10
    return accuracy

def classifyModel(X_train_cv, X_test_cv, y_train, y_test):
    # create model
    # used decision tree because it's multiclass
    dt = DecisionTreeClassifier()
    dt.fit(X_train_cv, y_train)
    # predict
    predictions = dt.predict(X_test_cv)
    # get accuracy
    accuracy = accuracy_score(y_test, predictions) * 100
    return accuracy

def main():
    lyrics, labels = loadData()
    # convert genres into numerical labels
    # numLabels is the new labels dataset
    index = 0
    dictLabels = {}
    numLabels = list()
    for label in labels:
        if label not in dictLabels:
            dictLabels[label] = index
            index += 1
    for x in range(len(labels)):
        numLabels.append(dictLabels[labels[x]])


    # shortening the lyrics & labels to test
    # will be removed in final
    lyricsShortened = list()
    labelsShortened = list()
    for x in range(100):
        lyricsShortened.append(lyrics[randint(1, len(lyrics))])
        labelsShortened.append(numLabels[randint(1, len(numLabels))])

    accuracy = makeClassifier(lyricsShortened, labelsShortened)
    #accuracy = makeClassifier(lyrics, numLabels)
    print("Accuracy: " + repr(accuracy) + "%")


main()
