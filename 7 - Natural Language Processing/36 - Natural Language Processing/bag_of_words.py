# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection._split
import sklearn.naive_bayes
import sklearn.feature_extraction.text
import nltk
import nltk.stem.porter
import nltk.corpus
import re

nltk.download('stopwords')

DATA_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv'
data = pd.read_csv(DATA_PATH, sep = '\t')
x, y = data.iloc[:, 0].values, data.iloc[:, 1].values

stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.porter.PorterStemmer()

def transform_review_text(text):
    text = text.lower()
    text = re.sub('[^a-z ]', '', text)
    text_array = [stemmer.stem(word) for word in text.split() if word not in stopwords]
    return ' '.join(text_array)
    
x = [transform_review_text(review) for review in x]

vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_features = 1500)
x = vectorizer.fit_transform(x).toarray()

