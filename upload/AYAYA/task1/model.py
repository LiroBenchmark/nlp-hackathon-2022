from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForTokenClassification
from tqdm import tqdm
from nervaluate import Evaluator

import re
from sklearn.metrics import accuracy_score
import numpy as np

import random


import torch
import torch.nn as nn

# import jax
# jax.random.PRNGKey(seed)

from unidecode import unidecode

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import json
from sklearn.utils.class_weight import compute_class_weight

class MyModel():
  def __init__(self):
    # do here any initializations you require
    # addings seeds
    seed = 8
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)

    # Defaults
    self.NUM_CLASSES = 31
    self.MAX_LENGTH = 128
    self.EPOCHS = 15
    self.BATCH_SIZE = 64
    self.NUM_LAYERS_FROZEN = 6
    self.LEARNING_RATE = 0.0001
    self.tags = ["O", "PERSON", "ORG", "GPE", "LOC", "NAT_REL_POL", "EVENT", "LANGUAGE", "WORK_OF_ART", "DATETIME", "PERIOD", "MONEY", "QUANTITY", "NUMERIC", "ORDINAL", "FACILITY"]

    self.classes = [ "O", "B-PERSON", "I-PERSON", "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-NAT_REL_POL", "I-NAT_REL_POL", "B-EVENT", "I-EVENT", "B-LANGUAGE",
                     "I-LANGUAGE", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-DATETIME", "I-DATETIME", "B-PERIOD", "I-PERIOD", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY",
                     "B-NUMERIC", "I-NUMERIC", "B-ORDINAL", "I-ORDINAL", "B-FACILITY", "I-FACILITY",
                    ]

    # train_data, train_labels = self.read_dataset(train_dataset, tokenizer=tokenizer)
    self.model = AutoModelForTokenClassification.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1", num_labels=self.NUM_CLASSES)
    self.tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

    # for param in self.model.bert.parameters():
    #   param.requires_grad = False

    for param in self.model.bert.embeddings.parameters():
      param.requires_grad = False
    for layer in self.model.bert.encoder.layer[:self.NUM_LAYERS_FROZEN]:
        for param in layer.parameters():
            param.requires_grad = False

  def load(self, model_resource_folder):
    self.model.load_state_dict(torch.load(model_resource_folder))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(device)

  def train(self, train_data_file, validation_data_file, model_resource_folder):
    # Prepairing the data
    print("Preparing data")
    X_train, y_train = self.read_dataset(train_data_file, tokenizer=self.tokenizer)
    X_val, y_val = self.read_dataset(validation_data_file, tokenizer=self.tokenizer)

    # Computing weights for our evaluation
    proper_labels = []
    for sequence in y_train:
        mini_label = []
        for label in sequence:
            if label != -100:
                proper_labels.append(int(label))
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(0, self.NUM_CLASSES), y=proper_labels)

    # Creating the datasets
    train_dataset = MyModel.MyDataset(X_train, y_train)
    validation_dataset = MyModel.MyDataset(X_val, y_val)

    # Creating the dataloaders
    train_dataloader = DataLoader(
      dataset=train_dataset,
      batch_size=self.BATCH_SIZE,
      shuffle=True
    )
    validation_dataloader = DataLoader(
      dataset=validation_dataset,
      batch_size=self.BATCH_SIZE
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move the model to GPU (when available)
    self.model.to(device)

    # create a SGD optimizer
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    # set up loss function
    loss_criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float).to(device))
    print("Training:")
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(1, self.EPOCHS + 1):
        train_loss, train_accuracy, pepega_labels = self.train_epoch(self.model, train_dataloader, loss_criterion, optimizer, device)
        val_loss, val_accuracy, pepega_pepega_labels = self.eval_epoch(self.model, validation_dataloader, loss_criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print('\nEpoch %d'%(epoch))
        print('train loss: %10.8f, accuracy: %10.8f'%(train_loss, train_accuracy))
        print('val loss: %10.8f, accuracy: %10.8f\n'%(val_loss, val_accuracy))

        torch.save(self.model.state_dict(), model_resource_folder)

  def predict(self, test_data_file):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      X_test, y_test, item_lengths = self.read_dataset(test_data_file, tokenizer=self.tokenizer, return_lengths=True)
      test_dataset = MyModel.MyDataset(X_test, y_test)

      test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=self.BATCH_SIZE
      )
      all_predictions, all_labels = self.test_epoch(self.model, test_dataloader, device)
      for idx in range(len(all_predictions)):
          all_predictions[idx] = self.classes[all_predictions[idx]]
          all_labels[idx] = self.classes[all_labels[idx]]
      reshaped_predictions = []
      reshaped_labels = []
      index = 0
      for value in item_lengths:
          reshaped_predictions.append(all_predictions[index: index + value])
          reshaped_labels.append(all_labels[index: index + value])
          index += value

      evaluator = Evaluator(reshaped_labels, reshaped_predictions, tags=self.tags, loader="list")
      results, results_by_tag = evaluator.evaluate()
      return results['strict']['f1']


  def train_epoch(self, model, train_dataloader, loss_crt, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = len(train_dataloader)
    predictions = []
    labels = []
    for idx, batch in tqdm(enumerate(train_dataloader)):
        batch_data, batch_labels = batch
        sequence_ids = batch_data['input_ids'].to(device, dtype=torch.long)
        sequence_masks = batch_data['attention_mask'].to(device)
        batch_labels = batch_labels.to(device)

        raw_output = model(input_ids=sequence_ids, attention_mask=sequence_masks, labels=batch_labels)
        loss, output = raw_output['loss'], raw_output['logits']
        logits = output.view(-1, model.num_labels)
        batch_predictions = torch.argmax(logits, dim=1)

        proper_labels = batch_labels.view(-1) != -100
        loss = loss_crt(logits, batch_labels.view(-1))

        filtered_labels = torch.masked_select(batch_labels.view(-1), proper_labels)
        filtered_predictions = torch.masked_select(batch_predictions, proper_labels)

        labels += filtered_labels.squeeze().tolist()
        predictions += filtered_predictions.tolist()

        batch_acc = accuracy_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy())
        epoch_acc += batch_acc

        loss_scalar = loss.item()

        # if idx % 500 == 0:
        #     print(epoch_acc/(idx + 1))
        #     print(batch_predictions)

        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=10
        )

        model.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss_scalar

    epoch_loss = epoch_loss/num_batches
    epoch_acc = epoch_acc/num_batches

    return epoch_loss, epoch_acc, labels

  def eval_epoch(self, model, val_dataloader, loss_crt, device):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = len(val_dataloader)
    predictions = []
    labels = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_dataloader)):
            batch_data, batch_labels = batch
            sequence_ids = batch_data['input_ids'].to(device, dtype=torch.long)
            sequence_masks = batch_data['attention_mask'].to(device)
            batch_labels = batch_labels.to(device)

            raw_output = model(input_ids=sequence_ids, attention_mask=sequence_masks, labels=batch_labels)
            loss, output = raw_output['loss'], raw_output['logits']
            logits = output.view(-1, model.num_labels)
            batch_predictions = torch.argmax(logits, dim=1)

            proper_labels = batch_labels.view(-1) != -100

            filtered_labels = torch.masked_select(batch_labels.view(-1), proper_labels)
            filtered_predictions = torch.masked_select(batch_predictions, proper_labels)

            labels += filtered_labels.squeeze().tolist()
            predictions += filtered_predictions.tolist()

            batch_acc = accuracy_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy())
            epoch_acc += batch_acc

            loss_scalar = loss.item()

            epoch_loss += loss_scalar

    epoch_loss = epoch_loss/num_batches
    epoch_acc = epoch_acc/num_batches

    return epoch_loss, epoch_acc, labels

  def test_epoch(self, model, test_dataloader, device):
    model.eval()
    epoch_loss = 0.0
    num_batches = len(test_dataloader)
    predictions = []
    labels = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dataloader)):
            batch_data, batch_labels = batch
            sequence_ids = batch_data['input_ids'].to(device, dtype=torch.long)
            sequence_masks = batch_data['attention_mask'].to(device)
            batch_labels = batch_labels.to(device)

            offset_mapping = batch_data['offset_mapping']

            raw_output = model(input_ids=sequence_ids, attention_mask=sequence_masks)
            output =  raw_output['logits']
            logits = output.view(-1, model.num_labels)
            batch_predictions = torch.argmax(logits, dim=1)


            filtered_predictions = []
            proper_labels = batch_labels.view(-1) != -100
            filtered_labels = torch.masked_select(batch_labels.view(-1), proper_labels)
            # print(f'len(filtered_labels): {len(filtered_labels)}')
            labels += filtered_labels.squeeze().tolist()

            for index, offset in enumerate(offset_mapping.view(-1, 2)):
                if offset[0] == 0 and offset[1] != 0:
                    filtered_predictions.append(batch_predictions[index].item())
            # print(f'len(filtered_predictions): {len(filtered_predictions)}')
            predictions += filtered_predictions

    return predictions, labels

  def get_tokens(self, dataset):
    token_list = ['0' for i in range(self.NUM_CLASSES)]
    for item in dataset:
        for id_ner_id in range(len(item['ner_ids'])):
            token_list[int(item['ner_ids'][id_ner_id])] = item['ner_tags'][id_ner_id]
    return token_list

  def read_dataset(self, dataset, tokenizer, train=True, return_lengths=False):
    data = []
    labels = []
    max_length = 0
    reshaped_dataset = []
    reshaped_labels = []
    item_lengths = []
    reshaped_length = 110
    for item in dataset:
        prelucrate_item = []
        item_lengths.append(len(item['ner_ids']))

        for token in item['tokens']:
            prelucrate_item.append(re.sub(r"\W+", 'n', token))

        for i in range(0, len(prelucrate_item), reshaped_length):
            reshaped_dataset.append(prelucrate_item[i: min(i + reshaped_length, len(prelucrate_item))])
            # print(item.keys())
            reshaped_labels.append( item['ner_ids'][i: min(i + reshaped_length, len(item['ner_ids']))])

    for index in range(len(reshaped_dataset)):
        items, sequence_labels =  reshaped_dataset[index], reshaped_labels[index]
        sequence = tokenizer(
            items,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_offsets_mapping=True

        )
        sequence = {key: torch.as_tensor(value) for key, value in sequence.items()}
        data.append(sequence)

        if train:
            encoded_labels = np.ones(len(sequence["offset_mapping"]), dtype=int) * -100
            # set only labels whose first offset position is 0 and the second is not 0
            i = 0
            for idx, offsets in enumerate(sequence["offset_mapping"]):
                if offsets[0] == 0 and offsets[1] != 0:
                    # overwrite label
                    encoded_labels[idx] = sequence_labels[i]
                    i += 1

            # max_length = max(len(sequence), max_length)
            labels.append(torch.as_tensor(encoded_labels))
    # print(max_length)
    if train:
        if return_lengths:
            return data, labels, item_lengths
        return data, labels

    return data

  class MyDataset(Dataset):
      def __init__(self, data, labels):
          super().__init__()
          self.data = data
          self.labels = labels

      def __getitem__(self, index):
          return self.data[index], self.labels[index]

      def __len__(self):
          return len(self.labels)

  class TestDataset(Dataset):
      def __init__(self, data, labels):
          super().__init__()
          self.data = data

      def __getitem__(self, index):
          return self.data[index]

      def __len__(self):
          return len(self.data)
