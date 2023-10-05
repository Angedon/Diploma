import numpy as np
from keras import models
from keras import layers
import os
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


dir = r"C:\Users\Gleb1\Desktop\Comments\Universities"


# Deleting tags in text
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


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def freq_code(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    res = list(vectorizer.get_feature_names_out())
    code = []
    k = 0
    for t in texts:
        arr = t.split(' ')
        p = []
        for j in range(len(arr) - 1, -1, -1):
            p.append(X[k, res.index(arr[j])])
        p.reverse()
        code.append(p)
        k += 1
    return code


data = pd.read_csv(f"{dir}{os.sep}all.csv")
text_data = list(data.Text)
text_marks = list(data.Mark)
cleared = []
for t in text_data:
    cleared.append(clear_text(t))
print(cleared[0])
code = freq_code(cleared)
#data_prepared = freq_code(text_data)
#data = vectorize(text_data)
#targets = np.array(text_marks).astype("float32")
train_x, train_y, test_x, test_y = code[:int(0.8*len(code))], targets[:int(0.8*len(code))], code[int(0.8*len(code)):], targets[int(0.8*len(code)):]
tokens = [set(word_tokenize(entry)) for entry in data['Text']]

#all_tokens = set()
#for t in tokens:
#    all_tokens = all_tokens.union(t)
#print(len(all_tokens))

model = models.Sequential()
# Input layer
model.add(layers.Dense(10, activation = "relu", input_shape=(10000, )))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(10, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(10, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
#model.summary()
model.compile(
optimizer = "adam",
loss = "binary_crossentropy",
metrics = ["accuracy"]
)
results = model.fit(
train_x, train_y,
epochs= 2,
batch_size = 32,
validation_data = (test_x, test_y)
)
scores = model.evaluate(test_x, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))