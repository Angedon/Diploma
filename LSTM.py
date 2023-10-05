import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
#import xgboost as xgb
seed = 4353

dir = r'C:\Users\Gleb1\Desktop\Comments\Universities'

#show any color
#plot_colortable(mcolors.CSS4_COLORS)

def change(mark):
    if mark == 'negative':
        return -1
    elif mark == 'positive':
        return 1
    else:
        return 0


ma = pymorphy2.MorphAnalyzer()
def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
    text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)
    #text = text.encode("utf-8")

    return text


def load_data_from_arrays(strings, labels, train_test_split=0.9):
    data_size = len(strings)
    test_size = int(data_size - round(data_size * train_test_split))
    print("Test size: {}".format(test_size))

    print("\nTraining set:")
    x_train = strings[test_size:]
    print("\t - x_train: {}".format(len(x_train)))
    y_train = labels[test_size:]
    print("\t - y_train: {}".format(len(y_train)))

    print("\nTesting set:")
    x_test = strings[:test_size]
    print("\t - x_test: {}".format(len(x_test)))
    y_test = labels[:test_size]
    print("\t - y_test: {}".format(len(y_test)))

    return x_train, y_train, x_test, y_test


data = pd.read_csv(dir + os.sep + 'all.csv')
print(data.head())

#data['Mark'] = data.apply(lambda x: change(x['Mark']), axis=1)
data['Mark'] = data.Mark.replace({'positive': 1, 'negative': -1, 'medium': 0})

print(data.head())

data.columns = data.columns.str.lower()
data.columns

print(data.mark[:10])
print(data['mark'].value_counts())
#sns.countplot(data['mark'].value_counts())
#plt.hist([-1, 0, 1], data['mark'].value_counts())
labels, counts = np.unique(data.mark, return_counts=True)
#plt.bar(labels, counts, align='center', color=['cornflowerblue', 'sandybrown', 'indianred'])
#plt.gca().set_xticks(labels)
#plt.xlabel('Marks')
#plt.ylabel('Count')
#plt.show()


data['text'] = data.apply(lambda x: clean_text(x['text']), axis=1)
text = data['text']
target = data['mark']

# создаем единый словарь (слово -> число) для преобразования
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text.tolist())

# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
textSequences = tokenizer.texts_to_sequences(text.tolist())

X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, target, train_test_split=0.8)

embedding_size = 32
max_words = 1000

model = Sequential()
model.add(Embedding(max_words, embedding_size, input_length=X_train.shape[1]))
model.add(LSTM(100))
model.add(Dense(3,activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# converting categorical variables in y_train to numerical variables
y_train_dummies = pd.get_dummies(y_train).values
y_test_dummies = pd.get_dummies(y_test).values
print('Shape of Label tensor: ', y_train_dummies.shape)

model.fit(X_train, y_train_dummies, epochs=5, batch_size=32)

model = load_model('MusicalInstrumentReviews.h5')
scores = model.evaluate(X_test, y_test_dummies)

LSTM_accuracy = scores[1]*100

print('Test accuracy: ', scores[1]*100, '%')

