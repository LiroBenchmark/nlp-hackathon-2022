# -*- coding: utf-8 -*-
"""SemanticTextSimilarityClassifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bh8IvbVd1r_TD0Zj6L9L7T9bz52dc1so
"""

!pip install sentence-transformers

# standard library imports
import logging
import os
import warnings
from datetime import datetime
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, LoggingHandler, models, losses, InputExample
from sentence_transformers.evaluation import TripletEvaluator, BinaryClassificationEvaluator

from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

# setup data
!wget -q https://raw.githubusercontent.com/dumitrescustefan/RO-STS/master/dataset/text-similarity/RO-STS.train.tsv
!wget -q https://raw.githubusercontent.com/dumitrescustefan/RO-STS/master/dataset/text-similarity/RO-STS.dev.tsv
!wget -q https://raw.githubusercontent.com/dumitrescustefan/RO-STS/master/dataset/text-similarity/RO-STS.test.tsv

train_df = pd.read_csv('RO-STS.train.tsv', sep='\t', names=['score', 'text_a', 'text_b'])
valid_df = pd.read_csv('RO-STS.dev.tsv', sep='\t', names=['score', 'text_a', 'text_b'])
test_df = pd.read_csv('RO-STS.test.tsv', sep='\t', names=['score', 'text_a', 'text_b'])

class MNRLDataset(IterableDataset):
    def __init__(self, clustered_sentences):
        self.corpus = clustered_sentences

    def __iter__(self):
        return iter(self.corpus)

    def __getitem__(self, idx):
        return self.corpus[idx]

    def __len__(self):
        return len(self.corpus)

class BertClassifierHelper:
    def __init__(self):
        self.raw_dataset = None
        self.dataset_final_form = []
        # self.first_form_dataset = []

    def __read_tsv_dataset_file(self):
        self.raw_dataset = train_df

    def __preprocess_sentence(self, sentence):
        return sentence.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")

    def __create_readable_dataset_format(self):
        for idx, row in self.raw_dataset.iterrows():
            self.dataset_final_form.append(InputExample(
                # row[0] -> score
                # row[1] -> first sentence
                # row[2] -> second sentence
                texts=[self.__preprocess_sentence(row[1]), self.__preprocess_sentence(row[2])],
                label=float(row[0]) / 5
            ))

    def __compute_training_dataset(self):
        return MNRLDataset(self.dataset_final_form)
        
    def create_training_dataset(self):
        self.__read_tsv_dataset_file()
        self.__create_readable_dataset_format()
        return self.__compute_training_dataset()

class MyModel():
  def __init__(self):
    self.name = 'MNRLBertClassifier'
    MODEL_NAME = 'dumitrescustefan/bert-base-romanian-uncased-v1'
    MAX_SEQ_LENGHT = 66

    # We construct the SentenceTransformer bi-encoder from scratch
    word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=MAX_SEQ_LENGHT)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)
    self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda')
    self.train_loss = losses.CosineSimilarityLoss(model=self.model)


  def load(self, model_resource_folder):
    self.model = SentenceTransformer(model_resource_folder)

  def train(self, train_data_file, validation_data_file, model_resource_folder):
    bertClassifierHelper = BertClassifierHelper()
    train_dataset = bertClassifierHelper.create_training_dataset()
    train_loader = DataLoader(train_dataset, batch_size=32, drop_last=True)

    logging.info("Train the model")
    epochs = 1
    warmup_steps = int(len(train_loader) * epochs * 0.1)
    self.model.fit(
        train_objectives=[(train_loader, self.train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=model_resource_folder,
        show_progress_bar=True,
        )

  def predict(self, test_data_file):
    sentences1 = []
    sentences2 = []
    for idx, row in test_data_file.iterrows():
      sentences1.append(row[1])
      sentences2.append(row[2])

    sentences = list(set(sentences1 + sentences2))
    embeddings = self.model.encode(sentences, batch_size=32, show_progress_bar=True,
                              convert_to_numpy=True)
    emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
    embeddings1 = [emb_dict[sent] for sent in sentences1]
    embeddings2 = [emb_dict[sent] for sent in sentences2]

    def cos_sim(emb1, emb2):
      return((emb1/np.linalg.norm(emb1)).dot(emb2/np.linalg.norm(emb2)))

    predicted_scores = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
      score = cos_sim(emb1, emb2) * 5
      predicted_scores.append(score)
    return test_data_file['score'].corr(pd.core.series.Series(predicted_scores), method='pearson')

    # we'll call this function after the load()
    # use this place to run the prediction
    # the output of this function is a single value, the Pearson correlation on the similarity score column of the test data and the predicted similiarity scores for each pair of texts in the test data.

# TRAINING
model = MyModel()
model.train(train_df, valid_df, "output")

# INFERENCE
from time import perf_counter 

# load model  
model = MyModel()
model.load("output")

# inference
start_time = perf_counter()
pearson_correlation = model.predict(test_df)
stop_time = perf_counter()

print("Predicted in {0}.".format(stop_time-start_time))
print("Pearson correlation = {0}".format(pearson_correlation))  # this is the score we want :)