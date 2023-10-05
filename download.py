import requests
import string
import csv
import os
#import preprocessing as prep

special = ["sfedu", "knitu", "sfu", "spbguap", "isu", "hse-nn", "gpmu", "szgmu", "ulspu", "timacad", "cchgeu", "sgu", "bashgu", "kubsau"]
dir = r"C:\Users\Gleb1\Desktop\Comments\Universities"
urls = [r"https://tabiturient.ru/vuzu/hse/",
       r"https://tabiturient.ru/vuzu/muctr/",
       r"https://tabiturient.ru/vuzu/urfu/",
       r"https://tabiturient.ru/vuzu/urfu/",
       r"https://tabiturient.ru/vuzu/mgtu/",
       r"https://tabiturient.ru/vuzu/spbstu/",
       r"https://tabiturient.ru/vuzu/sfu/",
       r"https://tabiturient.ru/vuzu/mirea/",
       r"https://tabiturient.ru/vuzu/kfu/",
       r"https://tabiturient.ru/vuzu/mai/",
       r"https://tabiturient.ru/vuzu/nsu/",
       r"https://tabiturient.ru/vuzu/spbgu/",
       r"https://tabiturient.ru/vuzu/mgu/",
       r"https://tabiturient.ru/vuzu/mipt/",
       r"https://tabiturient.ru/vuzu/eltech/",
       r"https://tabiturient.ru/vuzu/mtusi/",
       r"https://tabiturient.ru/vuzu/sut/",
       r"https://tabiturient.ru/vuzu/misis/",
       r"https://tabiturient.ru/vuzu/fu/",
       r"https://tabiturient.ru/vuzu/unn/",
       r"https://tabiturient.ru/vuzu/mgmu/",
       r"https://tabiturient.ru/vuzu/mgpu/",
       r"https://tabiturient.ru/vuzu/spmi/",
       r"https://tabiturient.ru/vuzu/ranepa",
       r"https://tabiturient.ru/vuzu/mgou/",
       r"https://tabiturient.ru/vuzu/miet/",
       r"https://tabiturient.ru/vuzu/gpmu/",
       r"https://tabiturient.ru/vuzu/spbgmu/",
       r"https://tabiturient.ru/vuzu/lgu_pushkin/",
       r"https://tabiturient.ru/vuzu/szgmu/",
       r"https://tabiturient.ru/vuzu/sziu_ranepa/",
       r"https://tabiturient.ru/vuzu/ulspu/",
       r"https://tabiturient.ru/vuzu/mgua/",
       r"https://tabiturient.ru/vuzu/timacad/",
       r"https://tabiturient.ru/vuzu/cchgeu/",
       r"https://tabiturient.ru/vuzu/herzen/",
       r"https://tabiturient.ru/vuzu/gubkin/",
       r"https://tabiturient.ru/vuzu/sgu/",
       r"https://tabiturient.ru/vuzu/bashgu/",
       r"https://tabiturient.ru/vuzu/mgupp/",
       r"https://tabiturient.ru/vuzu/mglu/",
       r"https://tabiturient.ru/vuzu/kubsau/",
       r"https://tabiturient.ru/vuzu/miet/",
       r"https://tabiturient.ru/vuzu/mtusi/",
       r"https://tabiturient.ru/vuzu/mgppu/"]

t_urls = [r"https://tabiturient.ru/vuzu/knitu/",
        r"https://tabiturient.ru/vuzu/sfedu/",
        r"https://tabiturient.ru/vuzu/dvfu/",
        r"https://tabiturient.ru/vuzu/spbguap/",
        r"https://tabiturient.ru/vuzu/isu/",
        r"https://tabiturient.ru/vuzu/hse-spb/",
        r"https://tabiturient.ru/vuzu/mgimo/",
        r"https://tabiturient.ru/vuzu/hse-nn/",
        r"https://tabiturient.ru/vuzu/miigaik/",
        r"https://tabiturient.ru/vuzu/mgppu/",
        r"https://tabiturient.ru/vuzu/rsuh/",
        r"https://tabiturient.ru/vuzu/guu/",
        r"https://tabiturient.ru/vuzu/mgou/",
        r"https://tabiturient.ru/vuzu/rguts/",
        r"https://tabiturient.ru/vuzu/stankin/"]


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
      writer.writerow({'Text': t[i], 'Mark': m[i]})


def read(dir, name):
  t, m = [], []
  with open(f'{dir}{os.sep}{name}.csv', 'r', newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      t.append(row['Text'])
      m.append(row['Mark'])
  return t, m


urls = urls + t_urls
k = 0
all_texts, all_marks = [], []
for u in urls:
  texts, all_text = foo(u)
  texts = delete_tags(texts)
  marks = form_marks(all_text, get_name(u))
  all_texts = all_texts + texts
  all_marks = all_marks + marks
  k += len(texts)
  print(f"url = {u}, len(marks) = {len(marks)}, len(texts) = {len(texts)}")
  write(texts, marks, get_name(u), dir)

write(all_texts, all_marks, "all", dir)
print("Count of all texts = ", k)



