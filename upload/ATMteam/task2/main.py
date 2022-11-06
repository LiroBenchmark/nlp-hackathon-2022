!wget -q https://raw.githubusercontent.com/Alegzandra/RED-Romanian-Emotions-Dataset/main/REDv2/data/train.json
!wget -q https://raw.githubusercontent.com/Alegzandra/RED-Romanian-Emotions-Dataset/main/REDv2/data/valid.json
!wget -q https://raw.githubusercontent.com/Alegzandra/RED-Romanian-Emotions-Dataset/main/REDv2/data/test.json
!pip install nlp
!pip install -q pandas sklearn

import pandas as pd
%matplotlib inline

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random
print('Using TensorFlow version', tf.__version__)
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

# let's install a couple of helpful packages needed for data exploration
df_train = pd.read_json('train.json')
df_test = pd.read_json('test.json')
df_val = pd.read_json('valid.json')

print("\nLet's see the columns: ")
print(df_train.columns)

print(f"\nWe have {len(df_train)} examples in the train dataset, {len(df_val)} in the validation and {len(df_test)} in the test sets.")

print("\nList some examples from the test dataset: ")
print(df_test[['text','agreed_labels']])

#['Tristețe', 'Surpriză', 'Frică', 'Furie', 'Neutru', 'Încredere', 'Bucurie']
print(df_train["text"])

class MyModel():
  def define_emotion(self, index):
    if index == 0:
      return "Tristete"
    elif index == 1:
      return "Surpriza"
    elif index == 2:
      return "Frica"
    elif index == 3:
      return "Furie"
    elif index == 4:
      return "Neutru"
    elif index == 5:
      return "Incredere"
    elif index == 6:
      return "Bucurie"

  def __init__(self):
    # do here any initializations you require
    emotions = [[]]
    n_elements = len(df_train)
    for i in range(n_elements):
      for j in range(7):
        if df_train[0]["agreed_labels"][i][j] == 1:
          emotions[i].append(self.define_emotion(j))
    

  def load(self, model_resource_folder):
    # we'll call this code before prediction
    # use this function to load any pretrained model and any other resource, from the given folder path
    model = torch.load(model_resource_folder)
    return model

  def train(self, train_json_file, validation_json_file, model_resource_folder):
    # we'll call this function right after init
    # place here all your training code
    # at the end of training, place all required resources, trained model, etc in the given model_resource_folder
    b=4


  def predict(self, test_json_file):
    # we'll call this function after the load()
    # use this place to run the prediction
    # the output of this function is a single value, the Hamming loss on the given json file
    a=3


# training phase
train_json = df_train
valid_json = df_val
test_json = df_test

model = MyModel()
model.train(train_json, valid_json, "output")

# right after we'll reload your model from the output folder
model = MyModel()
model.load("output")
hamming_loss = model.predict(test_json)
print(f"Hamming loss = {hamming_loss}")

import numpy as np
from sklearn.metrics import hamming_loss

y_gold =          np.array([[   1, 0, 0, 0, 0, 0, 1], [0,   0,   1, 0, 0, 0,   0]])
y_pred =          np.array([[0.33, 1, 0, 0, 0, 0, 0], [0, 0.1, 0.8, 0, 0, 0, 0.1]])
y_pred_opposite = np.array([[   0, 1, 1, 1, 1, 1, 0], [1,   1,   0, 1, 1, 1,   1]])
y_pred_perfect =  np.array([[   1, 0, 0, 0, 0, 0, 1], [0,   0,   1, 0, 0, 0,   0]])

y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
h_loss = hamming_loss(y_gold, y_pred)
print(f"Hamming loss = {h_loss}") # should print 0.21428571428571427

h_loss = hamming_loss(y_gold, y_pred_opposite)
print(f"Opposite Hamming loss = {h_loss}") # should print 1.0

h_loss = hamming_loss(y_gold, y_pred_perfect)
print(f"Perfect Hamming loss = {h_loss}") # should print 0.0