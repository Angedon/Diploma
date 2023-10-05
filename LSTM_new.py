import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import string
from io import BytesIO
from tokenize import tokenize
from collections import Counter
from tokenize import tokenize
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.probability import FreqDist
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
import pymorphy2

dir = r'C:\Users\Gleb1\Desktop\Comments\Universities'
nltk.download('stopwords')
from nltk.corpus import stopwords

ma = pymorphy2.MorphAnalyzer()


class Text:
    def __init__(self, texts, targets):
        self.targets = targets
        self.texts = [self.clear_text(t) for t in texts]
        self.arr_text = self.get_arr(self.texts)
        self.all_text = self.get_all_text(self.arr_text)
        self.words = self.get_words(self.all_text)
        #self.encoded_text = self.get_encoded()
        self.stop_list = self.get_stop_list()

    @staticmethod
    def get_all_text(arr):
        all_arr = np.array([])
        for e in arr:
            all_arr = np.concatenate([all_arr, e], axis=0)
        return all_arr

    @staticmethod
    def get_words(arr):
        words = Counter(arr)
        words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1])}
        return words

    def get_arr(self, texts):
        arr = []
        for t in texts:
            text = np.array(t.split())
            for w in self.get_stop_list():
                arr.append(text[text != w])
        return np.array(arr)

    def get_encoded(self):
        keys = np.array(list(self.words.keys()))
        keys = np.flip(keys)
        encoded = []
        for line in self.arr_text:
            buf = []
            for word in line:
                buf.append(np.where(keys == word)[0][0]+1)
            encoded.append(buf)
        return encoded

    def get_words_freq(self):
        return np.flip(np.array(list(self.words.keys())))

    @staticmethod
    def delete_tag(s):
        p = ""
        i = 0
        while i < len(s):
            if s[i] == '<':
                while s[i] != '>':
                    i += 1
                i += 1
            else:
                p += s[i]
                i += 1
        return p

    def clear_text(self, text):
        text = self.delete_tag(text)
        text = text.lower()
        special = [c for c in '[@_!#$%^&*()<>?/\|}{~:]']
        punctuations = [c for c in string.punctuation]
        russian_letters = [c for c in "абвгдеёжзийклмнопрстуфхцчшщыъьэюя"]
        english_letters = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        numbers = [c for c in "0123456789"]
        s = ""
        for i in range(len(text)):
            if (text[i] == ' ') or (text[i] in russian_letters):
                s += text[i]
            else:
                s += ' '
        s = " ".join(ma.parse(word)[0].normal_form for word in s.split())
        s = ' '.join(word for word in s.split() if len(word) > 3)
        return s.replace("  ", " ")

    @staticmethod
    def get_stop_list():
        stop_list = set(stopwords.words('russian'))
        with open(dir + os.sep + 'stop_words_russian.txt') as f:
            s = f.read()
        for word in s.split('\n'):
            stop_list.add(word)
        stop_list.add('ни')
        stop_list.add('не')
        stop_list.add('но')
        stop_list.add('ну')
        stop_list.add('на')
        return stop_list


def thread_work(text, target, i, j):
    return Text(list(text[i:j]), list(target[i:j]))


def analyze(t):
    tokenizer = Tokenizer(num_words=1000)
    arr = t.all_text
    texts = []
    for s in arr:
        texts.append(" ".join(s))
    texts = tokenizer.sequences_to_matrix(texts, mode='binary')
    return texts


df = pd.read_csv(dir + os.sep + 'all.csv')
texts = list(df['Text'])
targets = list(df['Mark'])
text = ['В некоторых примерах код при запуске может работать без исключений, но может содержать логические ошибки из-за использования переменных не того типа. А в некоторых примерах он может даже не выполняться.']
#t = [Text(list(df['Text'][:30]), list(df['Mark'][:30])), Text(list(df['Text'][30:60]), list(df['Mark'][30:60])), Text(list(df['Text'][60:90]), list(df['Mark'][60:90]))]
#t = Text(texts, targets)
t = Text(texts, targets)
arr = analyze(t)
print(arr[0])

