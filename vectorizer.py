import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import string


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


def delete_tags(texts):
  result = []
  for text in texts:
    result.append(delete_tag(text))
  return result


def clear_text(text):
    text = delete_tag(text)
    punctuations = [c for c in string.punctuation]
    english_letters = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    numbers = [c for c in "0123456789"]
    for i in range(len(text)):
      if (text[i] in punctuations) | (text[i] in english_letters) | (text[i] in numbers):
        text = text.replace(text[i], " ")
    return text.lower()


dir = r"C:\Users\Gleb1\Desktop\Comments\Universities"
dic = {}
data = pd.read_csv(f"{dir}{os.sep}all.csv")
txt = list(data['Text'])
targets = list(data['Mark'])
texts = []
for t in txt:
    texts.append(clear_text(t))

for t in texts[:int(len(texts)/2)]:
    words = t.split(' ')
    for w1 in words:
        if w1 not in dic:
            dic[w1] = {}
        for w2 in words:
            dic[w1][w2] = 1


df = pd.DataFrame(dic)
df.fillna(0, inplace=True)

print(df.shape)

pca = PCA()
res = pca.fit_transform(df)
plt.figure(figsize=(7, 7))
plt.scatter(res[:,0], res[:,1])
for i, label in enumerate(df.columns):
    x, y = res[i, 0], res[i, 1]
    plt.scatter(x, y)
    #kek = {'has': (1, 50), 'is': (1, 5)}
    plt.annotate(label, xy=(x, y), textcoords='offset points',
                   ha='right', va='bottom', )