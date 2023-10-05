from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd
import os

def clear_text(text):
  text = delete_tag(text)
  punctuations = [c for c in string.punctuation]
  english_letters = [c for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]
  numbers = [c for c in "0123456789"]
  for i in range(len(text)):
    if (text[i] in punctuations) | (text[i] in english_letters) | (text[i] in numbers):
      text = text.replace(text[i], " ")


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


def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN):
    train_InputExamples = train.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    validation_InputExamples = test.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    return train_InputExamples, validation_InputExamples

    train_InputExamples, validation_InputExamples = convert_data_to_examples(train,
                                                                             test,
                                                                             'DATA_COLUMN',
                                                                             'LABEL_COLUMN')


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []  # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,  # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,  # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                                                     input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = 'Text'
LABEL_COLUMN = 'Mark'


model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=50, validation_split=0.1,
    subset='training', seed=123)
test = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=50, validation_split=0.1,
    subset='validation', seed=123)

for i in train.take(1):
  train_feat = i[0].numpy()
  train_lab = i[1].numpy()

train = pd.DataFrame([train_feat, train_lab]).T
train.columns = ['Text', 'Mark']
train['Text'] = train['Text'].str.decode("utf-8")

for j in test.take(1):
  test_feat = j[0].numpy()
  test_lab = j[1].numpy()

test = pd.DataFrame([test_feat, test_lab]).T
test.columns = ['Text', 'Mark']
test['Text'] = test['Text'].str.decode("utf-8")

train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(50).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs=2, validation_data=validation_data)
print(f"Score = {model.score}")