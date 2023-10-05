import requests
import os
import csv

special = ["sfedu", "knitu", "sfu", "spbguap", "isu", "hse-nn"]


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

def foo(url):
  #'<div style="text-align:justify;" class="font2">'
  headStart = ['<div', 'style="text-align:justify;"', 'class="font2">']
  headEnd = '</div>'
  #s = requests.get(url='https://tabiturient.ru/vuzu/sut/').text
  s = requests.get(url=url).text
  headsIndicesStart, headsIndicesEnd, texts = [], [], []
  #print(s)
  k = 0
  indStart = 0
  indEnd = 1
  buff = 0
  while(True):
    buff = s.find(headStart[0], buff, len(s))
    if (s[buff+5:buff+32] == headStart[1]) & (s[buff+33 : buff+47] == headStart[2]):
      indStart = buff
      indEnd = s.find(headEnd, indStart + 47, len(s))
      headsIndicesStart.append(indStart)
      headsIndicesEnd.append(indEnd)
      k += 1
    if buff == -1:
      break
    buff += 4

  for i in range(k):
    left, right = headsIndicesStart[i], headsIndicesEnd[i]+len(headEnd)
    texts.append(s[left:right])
  return texts, s


def get_name(url):
  i = len(url)-2
  while url[i] != "/":
    i -= 1
  return url[(i+1):len(url)-1]


def form_marks(t, name):
  i = 0
  indices = []
  survey = []
  find = r'<img src="https://tabiturient.ru/img/smile'
  if name not in special:
    i = t.find('<img src="https://tabiturient.ru/img/smile', i) + 1
    i = t.find('<img src="https://tabiturient.ru/img/smile', i) + 1
    i = t.find('<img src="https://tabiturient.ru/img/smile', i) + 1
  while True:
    i = t.find('<img src="https://tabiturient.ru/img/smile', i)
    #print(t[i:(i+10)])
    if i == -1:
      break
    else:
      indices.append(i)
      if t[i + len(find)] == '1':
        survey.append('positive')
      elif t[i + len(find)] == '2':
        survey.append('negative')
      elif t[i + len(find)] == '3':
        survey.append('medium')
      i += 1
  return survey


def write(t, m, name, dir):
  with open(f'{dir}{os.sep}{name}.csv', 'w', newline='', encoding="utf-8") as csvfile:
    fieldnames = ['Text', 'Mark']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(t)):
      writer.writerow({'Text': t[i], '__label__': m[i]})


def read(dir, name):
  t, m = [], []
  with open(f'{dir}{os.sep}{name}.csv', 'r', newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      t.append(row['Text'])
      m.append(row['Mark'])
  return t, m


def write_txt(t, m, name, dir):
    with open(f'{dir}{os.sep}{name}.txt', 'w', newline='', encoding="utf-8") as f:
        for i in range(len(t)):
            text = t[i].replace('\n', ' ')
            f.write(f"{text} __label__={m[i]}\n")