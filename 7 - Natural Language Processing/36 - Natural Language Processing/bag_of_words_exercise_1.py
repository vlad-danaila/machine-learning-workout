# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.feature_extraction.text
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.naive_bayes
import nltk as nl
import re
import nltk.corpus
import nltk.stem.porter

# Data loading
DATA_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv'
data = pd.read_csv(DATA_PATH, sep = '\t')

# Stop words
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Stemmer
stemmer = nltk.stem.PorterStemmer()

# Preprocess data
x, y = [], []
for review in data.values:
    text = review[0]
    text = text.lower()
    text = re.sub('[0-9.,!?/\"\']', '', text)
    text = text.split(' ')
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    x.append(' '.join(text))
    y.append(review[1])
    
# Split train - test
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y)

# Vectorizer
vectorizer = sk.feature_extraction.text.CountVectorizer()
x_train = vectorizer.fit_transform(x_train).toarray()
x_test = vectorizer.transform(x_test).toarray()

# Define model
classifier = sk.naive_bayes.MultinomialNB()
classifier.fit(x_train, y_train)

# Predict
y_pred = classifier.predict(x_test)
conf_matrix = sk.metrics.confusion_matrix(y_pred, y_test)
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)