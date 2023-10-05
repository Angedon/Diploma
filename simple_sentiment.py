import random
import os
import pandas as pd
import numpy as np
import spacy
import csv
import sklearn.metrics
from spacy.util import minibatch, compounding

dir = r"C:\Users\Gleb1\Desktop\Comments\Universities"


def write(t, m, name, dir):
  with open(f'{dir}{os.sep}{name}.csv', 'w', newline='', encoding="utf-8") as csvfile:
    fieldnames = ['Text', 'Mark']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(t)):
      writer.writerow({'Text': t[i], 'Mark': m[i]})


def write_txt(t, m, name, dir):
    with open(f'{dir}{os.sep}{name}.txt', 'w', newline='', encoding="utf-8") as f:
        for i in range(len(t)):
            f.write(f"{t[i]} __label__={m[i]}\n")


def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    # Указываем TP как малое число, чтобы в знаменателе
    # не оказался 0
    TP, FP, TN, FN = 1e-8, 0, 0, 0
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        score_pos = review.cats['pos']
        if true_label['pos']:
            if score_pos >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if score_pos >= 0.5:
                FP += 1
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


def train_model(
        training_data: list,
        test_data: list,
        iterations: int = 20) -> None:
    # Строим конвейер
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Обучаем только textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Начинаем обучение")
        print("Loss\t\tPrec.\tRec.\tF-score")  #
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # Генератор бесконечной последовательности входных чисел
        for i in range(iterations):
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(
                    text,
                    labels,
                    drop=0.2,
                    sgd=optimizer,
                    losses=loss
                )
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(  #
                    tokenizer=nlp.tokenizer,  #
                    textcat=textcat,  #
                    test_data=test_data  #
                )  #
                print(f"{loss['textcat']:9.6f}\t\
                        {evaluation_results['precision']:.3f}\t\
                        {evaluation_results['recall']:.3f}\t\
                        {evaluation_results['f-score']:.3f}")

    # Сохраняем модель                                 #
    with nlp.use_params(optimizer.averages):  #
        nlp.to_disk("model_artifacts")  #


datas = pd.DataFrame({})
k = []
for f in os.listdir(dir):
    data = pd.read_csv(dir + os.sep + f)
    k.append(len(data))
    data['Mark'].replace(['positive', 'negative', 'medium'], [1, -1, 0], inplace=True)
    if len(datas) != 0:
        datas = pd.concat([datas, data])
    else:
        datas = data

reviews = []
t = list(datas.Text)
m = list(datas.Mark.values)
for i in range(len(t)):
    space = {
        'cats':{
            'pos' : m[i] == 1,
            'neg' : m[i] == 0
        }
    }
    reviews.append((t[i], space))

#random.shuffle(reviews)
#print(reviews[0])
#split = int(len(reviews) * 0.9)
#train, test = reviews[:split], reviews[split:]
#train_model(train, test)
print(len(t), " ", len(m))
write(t, m, 'all', dir)
write_txt(t, m, 'all_t', dir)