from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import pandas as pd
import os

dir = r'C:\Users\Gleb1\Desktop\Comments\test'
tokenizer = RegexTokenizer()
df = pd.read_csv(dir + os.sep + 'all.csv')
m = list(df.Mark.values)
t = list(df.Text)
texts = []
for text in t:
    texts.append(" ".join(line.strip() for line in text.splitlines()))
print(texts[0])
model = FastTextSocialNetworkModel(tokenizer=tokenizer)
res = model.predict(texts, k=3)
for message, sentiment in zip(m, res):
    print(message, '-&gt;', sentiment)
print(model.predict(['Это был ужасный фильм.']))