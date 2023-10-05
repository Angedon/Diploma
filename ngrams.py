import string, nltk, collections
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.ru import Russian
from func import read

stemmer = SnowballStemmer("russian")
nlp = Russian()


def get_tokens(texts):
    tokens = []
    for t in texts:
        tokens += t.split()
    return tokens


def lemmatization(text):
   doc = nlp(text)
   for token in doc:
      print(token, token.lemma, token.lemma_)
   tokens = [token.lemma_ for token in doc]
   return " ".join(tokens)


dir = r'C:\Users\Gleb1\Desktop\Comments\Universities'
punc = [c for c in string.punctuation]
print(type(punc))
english_letters = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]
numbers = [c for c in "0123456789"]
odd = punc + english_letters + numbers

texts_train, marks_train = read(r'C:\Users\Gleb1\Desktop\Comments\Universities', 'all')
texts_test, marks_test = read(r'C:\Users\Gleb1\Desktop\Comments\test', 'all')
texts = texts_train + texts_test
marks = marks_train + marks_test

for i in range(len(texts)):
    for s in odd:
        texts[i] = texts[i].replace(s, "")

tokens = get_tokens(texts)
bigrams = ngrams(tokens, 2)
trigrams = ngrams(tokens, 3)
fourgrams = ngrams(tokens, 4)
bigram_freq = collections.Counter(bigrams)
trigram_freq = collections.Counter(trigrams)
fourgram_freq = collections.Counter(fourgrams)
#print(bigram_freq.most_common(15))
print(trigram_freq.most_common(35))
#print(fourgram_freq.most_common(15))
