import wandb

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from wandb.keras import WandbCallback

train_data = pd.read_csv("../../BigData/jigsaw/train.csv")
print(train_data.head())

X_train = train_data["comment_text"]
y_train = train_data.iloc[:, 2:]
y_train = y_train.values

# model versioning config
MODEL_NAME = "LSTM_baseline"
SAVE_PATH = "lstm_baseline"

# config
NUM_WORDS = 100000
EMBED_IN = 150000
EMBED_OUT = 300
L1 = 128
L2 = 32
KERNEL=3
DROPOUT=0.2

# tokenize
tokenizer = keras.preprocessing.text.Tokenizer(num_words = NUM_WORDS, oov_token='<oov>')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
maxlen = max([len(x) for x in np.array(X_train)])
print(maxlen)

X_train = keras.preprocessing.sequence.pad_sequences(X_train, padding="pre", truncating="pre", maxlen=maxlen)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

wandb.init(project="jigsaw")

cfg = {"num_train" : len(y_train), "num_val" : len(y_val), "vocab" : NUM_WORDS,
  "maxlen" : maxlen, "embed_in" : EMBED_IN, "embed_out" : EMBED_OUT, "l1": L1, "l2" : L2}
wandb.config.update(cfg)


model = tf.keras.Sequential([tf.keras.layers.Embedding(EMBED_IN, EMBED_OUT),
    tf.keras.layers.LSTM(L1, return_sequences = True),
    tf.keras.layers.Conv1D(filters=L1, kernel_size=KERNEL, padding='valid', kernel_initializer='glorot_uniform'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(L2, activation='relu'),
    tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(6, activation='sigmoid')])
callbacks = [WandbCallback(log_evaluation=True)]                             
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['AUC'])
model.fit(X_train, y_train, epochs = 2, batch_size = 200, callbacks=callbacks)

sample_text = ["fuck this shit I'm out"]
sample_text = tokenizer.texts_to_sequences(sample_text)
sample_text = keras.preprocessing.sequence.pad_sequences(sample_text, padding="pre", truncating="pre", maxlen=maxlen)
sample_prediction = model.predict(sample_text)

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for i in range(len(labels)):
    print(labels[i] + " = " + str(sample_prediction[0][i]))

model_at = wandb.Artifact(
             MODEL_NAME, type="lstm_model",
             description="lstm first pass", 
             metadata=dict(cfg))

# save model
model.save(SAVE_PATH)
model_at.add_file(SAVE_PATH)
wandb.run.log_artifact(model_at)
