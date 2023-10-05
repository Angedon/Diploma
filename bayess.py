import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import re
import func


def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','USER', text)
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +',' ', text)
    return text.strip()


def get_all_texts(t, m):
    positive = []
    negative = []
    medium = []
    for i in range(len(t)):
        if m[i] == 'positive':
            positive.append(t[i])
        elif m[i] == 'negative':
            negative.append(t[i])
        elif m[i] == 'medium':
            medium.append(t[i])
    return np.array(positive), np.array(negative), np.array(medium)


dir = r'C:\Users\Gleb1\Desktop\Comments\Universities'
t, m = func.read(dir, 'all')
data_positive, data_negative, data_medium = get_all_texts(t, m)
print(data_positive.shape[0], " ", data_negative.shape[0], " ", data_medium.shape[0])
sample_size = min(data_positive.shape[0], data_negative.shape[0])
sample_size = min(data_medium.shape[0], sample_size)
raw_data = np.concatenate((data_positive[:sample_size],
                           data_negative[:sample_size], data_medium[:sample_size]), axis=0)
labels = [1]*sample_size + [-1]*sample_size + [0]*sample_size

data = [preprocess_text(text) for text in raw_data]
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

print(x_train[0])
print(labels[0])
score = 'f1_macro'
print("# Tuning hyper-parameters for %s" % score)
np.errstate(divide='ignore')
clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring=score)
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for mean, std, params in zip(clf.cv_results_['mean_test_score'],
                             clf.cv_results_['std_test_score'],
                             clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
print(classification_report(y_test, clf.predict(x_test), digits=4))
print()