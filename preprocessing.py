import string
from nltk.corpus import stopwords
import nltk
import csv
import os
import pandas as pd
from gensim.models import Phrases
from gensim.models import Word2Vec
from tokenize import tokenize
from io import BytesIO
from Levenshtein import distance
from Levenshtein import ratio


dir = r"C:\Users\Gleb1\Desktop\Comments\Universities"
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")


def get_name(url):
  i = len(url)-2
  while url[i] != "/":
    i -= 1
  return url[(i+1):len(url)-1]


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


def read(dir, name):
  t, m = [], []
  with open(f'{dir}{os.sep}{name}.csv', 'r', newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      t.append(row['Text'])
      m.append(row['Mark'])
  return t, m


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


def clear_text(text):
  text = delete_tag(text)
  punctuations = [c for c in string.punctuation]
  english_letters = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]
  numbers = [c for c in "0123456789"]
  for i in range(len(text)):
    if (text[i] in punctuations) | (text[i] in english_letters) | (text[i] in numbers):
      text = text.replace(text[i], " ")

  #text = "".join([ch for ch in text if ch not in punctuations])
  #text = "".join([ch for ch in text if ch not in english_letters])
  #text = "".join([ch for ch in text if ch not in numbers])
  text = text.lower()
  #brr = text.split(" ")
  #crr = [brr[i] for i in range(len(brr)) if brr[i] not in russian_stopwords]
  #text = " ".join(crr)
  return text


t, m = read(dir, "mipt")
bigram = Phrases(t[0])
text = t[0]
df = pd.read_csv(f'{dir}{os.sep}{"mipt"}.csv')
print(df.head(5))
for text in t:
  clear_text(text)

model = Word2Vec(bigram[t[0]], min_count=1)
print(model.sample)
tokenSet = set()
words = ["общага", "общаг", "общаги", "общагу"]
find_words = {}
for text in t:
  tokens = tokenize(BytesIO(text.encode('utf-8')).readline)
  for i in tokens:
    tokenSet.add(i.string)
    for w in words:
      if(ratio(w, i.string) >= 0.75):
        if (i.string not in find_words.keys()):
          find_words[i.string] = 1
        else:
          find_words[i.string] += 1

print(find_words)
min = float('inf')
word = ""
for w in tokenSet:
  if distance(w, "дважды") < min:
    min = distance(w, "дважды")
    word = w
#print(word)
