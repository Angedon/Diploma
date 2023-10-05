import pandas as pd
import os
import string
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer


def clear_text(text):
  punctuations = [c for c in string.punctuation]
  english_letters = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]
  numbers = [c for c in "0123456789"]
  for i in range(len(text)):
    if (text[i] in punctuations) | (text[i] in english_letters) | (text[i] in numbers):
      text = text.replace(text[i], " ")

  text = text.lower()
  return text


def filter(tokens):
    filtered_tokens = []
    for w in tokens:
        if w not in stop_words:
            filtered_tokens.append(w)
    return filtered_tokens


tokenizer = RegexpTokenizer(r'\w+')
stop_words=set(stopwords.words("russian"))
dir = r"C:\Users\Gleb1\Desktop\Comments\Universities"

df = pd.read_csv(dir + os.sep + 'all.csv')
texts = list(df.Text)
marks = list(df.Mark.values)
print(df.Mark.values)
tokens_pos = []
tokens_med = []
tokens_neg = []
summary_symbols = [0, 0, 0]
counts = [0, 0, 0]
for i in range(len(texts)):
    text = clear_text(texts[i])
    if marks[i] == 1:
        counts[0] += 1
        summary_symbols[0] += len(text)
        buff = word_tokenize(text)
        tokens_pos = tokens_pos + buff
    elif marks[i] == 0:
        counts[1] += 1
        summary_symbols[1] += len(text)
        buff = word_tokenize(text)
        tokens_med = tokens_med + buff
    elif marks[i] == -1:
        counts[2] += 1
        summary_symbols[2] += len(text)
        buff = word_tokenize(text)
        tokens_neg = tokens_neg + buff

fdist = FreqDist(tokens_neg)

filtered_tokens_pos = filter(tokens_pos)
filtered_tokens_neg = filter(tokens_neg)
filtered_tokens_med = filter(tokens_med)

print(f"Количество позитивных отзывов = {counts[0]}")
print(f"Количество средних отзывов = {counts[1]}")
print(f"Количество негативных отзывов = {counts[2]}")

print(f"Количество токенов в позитивных отзывах = {len(tokens_pos)}")
print(f"Количество токенов в средних отзывах = {len(tokens_med)}")
print(f"Количество токенов в негативных отзывах = {len(tokens_neg)}")

fdist_pos = FreqDist(filtered_tokens_pos)
fdist_neg = FreqDist(filtered_tokens_neg)
fdist_med = FreqDist(filtered_tokens_med)

print(fdist_pos.most_common(10))
print(fdist_neg.most_common(10))
print(fdist_med.most_common(10))

print(f"В среднем {summary_symbols[0] / counts[0]} символов в позитивных комментариях.")
print(f"В среднем {summary_symbols[1] / counts[1]} символов в средних комментариях.")
print(f"В среднем {summary_symbols[2] / counts[2]} символов в негативных комментариях.")

set_neg = set(filtered_tokens_neg) - (set(filtered_tokens_pos) & set(filtered_tokens_neg) & set(filtered_tokens_med) | set(filtered_tokens_neg) & set(filtered_tokens_med) | set(filtered_tokens_neg) & set(filtered_tokens_pos))
f_neg = [f for f in filtered_tokens_neg if f in set_neg]
fdist_neg_unique = FreqDist(f_neg)

set_pos = set(filtered_tokens_pos) - (set(filtered_tokens_pos) & set(filtered_tokens_neg) & set(filtered_tokens_med) | set(filtered_tokens_pos) & set(filtered_tokens_med) | set(filtered_tokens_neg) & set(filtered_tokens_pos))
f_pos = [f for f in filtered_tokens_pos if f in set_pos]
fdist_pos_unique = FreqDist(f_pos)

set_med = set(filtered_tokens_med) - (set(filtered_tokens_pos) & set(filtered_tokens_neg) & set(filtered_tokens_med) | set(filtered_tokens_med) & set(filtered_tokens_neg) | set(filtered_tokens_med) & set(filtered_tokens_pos))
f_med = [f for f in filtered_tokens_med if f in set_med]
fdist_med_unique = FreqDist(f_med)
#print(fdist_neg_unique.most_common(10))

#fdist_pos_unique.plot(30, cumulative=False)
#fdist_med_unique.plot(30, cumulative=False)
#fdist_neg_unique.plot(30, cumulative=False)
print(fdist_pos_unique.most_common(10))
print(fdist_med_unique.most_common(10))
print(fdist_neg_unique.most_common(10))

#plt.show()
