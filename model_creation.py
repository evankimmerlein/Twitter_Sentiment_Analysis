##########################
# © 2022 Evan Kimmerrlein
#   All Rights Reserved
##########################
# Dataset intended to be used with this code can be found at https://www.kaggle.com/datasets/kazanova/sentiment140https://www.kaggle.com/datasets/kazanova/sentiment140
# IMPORTS

import pandas as pd
import numpy as np
import keras
import pickle

from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential


import matplotlib.pyplot as plt
from collections import Counter

import nltk
from nltk.corpus import stopwords
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import gensim
import time
import logging


# CONSTANTS

# Data
FEATURES = ["target", "ids", "date", "flag", "user", "text"]
ENCODING = "ISO-8859-1"


# Sentiment
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
POSITIVE = "POSITIVE"

# Paths
DATA_PATH = "data/input_cleaner.csv"


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# READ IN DATA

df = pd.read_csv(DATA_PATH, encoding =ENCODING , names=FEATURES, quotechar='"', engine='python', error_bad_lines=False)
def decode_sentiment(value):
    if value == 0:
        return "NEGATIVE"
    else:
        return "POSITIVE"
  
df.target = df.target.apply(lambda x: decode_sentiment(x))

target_cnt = Counter(df.target)

plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset Labels Count")



# DATA PREP

stop_words = stopwords.words("english")

def cleanText(text):

    text = re.sub('@\\S+|https?:\\S+|http?:\\S|[^A-Za-z0-9]+', ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            tokens.append(token)
    return " ".join(tokens)

df.text = df.text.apply(lambda x:cleanText(x))

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


# BUILD W2V

model = gensim.models.word2vec.Word2Vec(window=7, min_count=10, workers=8)

docs = [_text.split() for _text in df_train.text] 
model.build_vocab(docs)

model.train(docs, total_examples=len(docs), epochs=32)

model.save(r'models\m2v.genism')

# BUILD TOKENIZER

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)

vocab_size = len(tokenizer.word_index) + 1

with open(r'models\tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=300)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=300)

# ENCODE LABELS

labels = df_train.target.unique().tolist()
labels.append(NEUTRAL)

encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())

y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# EMBEDDING
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
  if word in model.wv:
    embedding_matrix[i] = model.wv[word]

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=300, trainable=False)

# BUILD SEQUENTIAL MODEL

model_seq = Sequential(
    [
        embedding_layer,
        Dropout(0.5),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ]
)

model_seq.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

model_seq_fit = model_seq.fit(x_train, y_train,
                    batch_size=1024,
                    epochs=8,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)])

model_seq.save(r"models\model.keras")

def score_to_sentiment(score):
    if score >= 0.5:
        return POSITIVE
    else:
        return NEGATIVE


def get_score(text):
  x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
  return model_seq.predict([x_test])[0][0]

# SHOW MODEL INFO

#Test Accuracy/Loss
acc_loss = model_seq.evaluate(x_test, y_test, batch_size=1024)
print(f"Accuracy {acc_loss[1]}:")
print(f"Loss:{acc_loss[0]}")

# Accuracy Graphs
acc = model_seq_fit.history['accuracy']
val_acc = model_seq_fit.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

# Loss Graphs
loss = model_seq_fit.history['loss']
val_loss = model_seq_fit.history['val_loss']
 
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()


# Confusion Matrix
y_cm_test = np.array([])
y_cm_pred = np.array([])

head_num = 10000
correct = 0
incorrect = 0

for entry in df_train.head(head_num).values:
  if(score_to_sentiment(get_score(entry[5])) == POSITIVE):
    y_cm_pred = np.append(y_cm_pred,1)
  else:
    y_cm_pred = np.append(y_cm_pred,0)
  if(entry[0] == POSITIVE):
    y_cm_test = np.append(y_cm_test,1)
  else:
    y_cm_test = np.append(y_cm_test,0)

cm_labels = ["Positive", "Negative"]
cm = confusion_matrix(y_cm_test, y_cm_pred, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()