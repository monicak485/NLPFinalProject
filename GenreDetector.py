# NLP Final Project
# Trigram Model to detect song genre based on lyrics
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
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
import pandas as pd

def loadData():
    with open('cleaned.csv', 'rt') as f:
        reader = csv.reader(f)
        songData = list(reader)
    labels = list()
    lyrics = list()

    for song in songData:
        labels.append(song[0])
        song[1].replace('\n', ' ')
        lyrics.append(song[1])

    return lyrics, labels

# finding the average accuracy in NBClassifier
def avgAccuracy(scores, count):
    cuts = float(len(scores))
    for num in scores:
        count = count + num
    result = count / cuts
    average = float(result)
    return average*100


def makeLRClassifier(data, labels):
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
        accuracy = classifyLRModel(tfidf_features, test_cv, trainLabels, testLabels)
        avgAccuracy += accuracy
    # get avg scores over all folds
    accuracy = avgAccuracy/10
    return accuracy

def makeNBClassifier(data, labels):
    # Columns that I will use for the separate dataframes:
    makeCols = ["Lyrics", "Labels"]
    # Put the columns in the dataframe
    dataframe = pd.DataFrame(columns = makeCols)
    # Set the Label column to the labels
    dataframe['Labels'] = labels
    # Set the Lyrics column to lyrics
    dataframe['Lyrics'] = data
    ly = dataframe['Lyrics']

    # A Naïve Bayes classifier with add-1 smoothing using binary bagof-words features
    # Code from the linked website for the assignment: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer()
    hello = vectorizer.fit_transform(ly)
    X = hello.toarray()
    Y = dataframe['Labels']

    # Naive Bayes Classifier code found from documentation: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    clf = MultinomialNB()
    clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    clf.fit(X, Y)
    cv_results = cross_validate(clf, X, Y, cv=10)

    # Need to find the accuracy for the 'test_score' part of the cv_results
    count = 0.00
    avg = avgAccuracy(cv_results['test_score'],count)
    return avg

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

def classifyLRModel(X_train_cv, X_test_cv, y_train, y_test):
    # creates model Logictic Regression Model
    dt = LogisticRegression(solver='lbfgs', multi_class = 'auto', max_iter = 1000)
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
    for x in range(1000):
        lyricsShortened.append(lyrics[randint(1, len(lyrics))])
        labelsShortened.append(numLabels[randint(1, len(numLabels))])

    accuracyClassifier1 = makeClassifier(lyricsShortened, labelsShortened)
    accuracyClassifier2 = makeNBClassifier(lyricsShortened, labelsShortened)
    accuracyClassifier3 = makeLRClassifier(lyricsShortened, labelsShortened)

    print("Accuracy for Decision Tree Classifier: " + repr(accuracyClassifier1) + "%")
    print("Accuracy for Naive Bayes Classifier: " + repr(accuracyClassifier2) + "%")
    print("Accuracy for Logistic Regression Classifier: " + repr(accuracyClassifier3) + "%")



main()
