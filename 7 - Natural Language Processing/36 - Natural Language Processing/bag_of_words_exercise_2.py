import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.feature_extraction
import sklearn.metrics
import nltk
import nltk.corpus
import nltk.stem
import pandas as pd
import matplotlib.pyplot as plt
import re

dataset_path = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv'

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

stemmer = nltk.stem.PorterStemmer()

vectorizer = sk.feature_extraction.text.CountVectorizer()

def process_text(text):
    text = re.sub('[^a-z ]', '', text.lower())
    return ' '.join([stemmer.stem(word) for word in text.split(' ') if word not in stopwords])

x, y = [], []

for line in open(dataset_path, 'r'):
    text, label = line.split('\t')
    text = process_text(text)
    x.append(text)
    y.append(label.strip())
    
x, y = x[1:], y[1:]
y = [int(_y) for _y in y]

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y)

x_train = vectorizer.fit_transform(x_train).toarray()
x_test = vectorizer.transform(x_test).toarray()

bayes = sk.naive_bayes.MultinomialNB()
bayes.fit(x_train, y_train)

y_pred = bayes.predict(x_test)
accuracy = sk.metrics.accuracy_score(y_test, y_pred)
print('Accuracy is', accuracy)