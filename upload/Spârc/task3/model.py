# Cred ca este nevoie sa fie instalata separat biblioteca `transformers`:
# !pip install transformers

import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModel


class ActualModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(2, 1)

  def forward(self, x):
    y = self.linear(x)
    return y


class MyModel():
  MODEL_FILENAME = "model.pth"

  def __init__(self):
    self.bert_tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
    self.bert_model = AutoModel.from_pretrained("readerbench/RoBERT-base")

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {self.device} device")
    self.model = ActualModel().to(self.device)
    print(self.model)

  def load(self, model_resource_folder):
    """Loads pre-trained model weights from a given folder."""
    model_path = MyModel.model_path(model_resource_folder)
    self.model.load_state_dict(torch.load(model_path))

  def train(self, train_df, valid_df, model_resource_folder):
    """Trains the model and saves it to disk."""
    train_dataloader = self.make_dataloader(train_df)
    valid_dataloader = self.make_dataloader(valid_df)

    loss_fn = self.loss
    optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    # Train the model.
    self.model.train()
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to(self.device), y.to(self.device)

        # Compute the prediction error.
        pred = self.model(x)
        loss = loss_fn(pred, y)

        # Run backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 25 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_df):>5d}]")

    # Save the trained model.
    self.save_to_disk(model_resource_folder)

  def predict(self, test_df):
    gold = test_df['score']

    # Get the predictions.
    self.model.eval()
    with torch.no_grad():
      inputs = self.prepare_inputs(test_df)
      pred = self.model(inputs)

    # Calculate the loss.
    pred = pd.Series(pred.cpu().numpy()[:, 0])
    return self.compute_correlation(pred, gold)

  def make_dataloader(self, df: pd.DataFrame) -> DataLoader:
    features = self.prepare_inputs(df)
    targets = torch.Tensor(df['score'].values).to(self.device)

    dataset = torch.utils.data.TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader

  def prepare_inputs(self, df: pd.DataFrame) -> torch.Tensor:
    # inputs = self.bert_tokenizer("exemplu de propoziție", return_tensors="pt")
    # outputs = model(**inputs)
    # return outputs.to(self.device)

    df = pd.DataFrame({
        'len_a': df['text_a'].map(len),
        'len_b': df['text_b'].map(len),
    })
    return torch.Tensor(df.values).to(self.device)

  def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    vpred = pred - torch.mean(pred)
    vy = y - torch.mean(y)
    return torch.sum(vpred * vy) / (torch.sqrt(torch.sum(vpred ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

  def compute_correlation(self, pred: pd.Series, gold: pd.Series) -> float:
    return gold.corr(pred, method='pearson')

  def save_to_disk(self, model_resource_folder: str):
    """Saves the trained model to disk."""
    os.makedirs(model_resource_folder, exist_ok=True)
    model_path = MyModel.model_path(model_resource_folder)
    torch.save(self.model.state_dict(), model_path)

  @classmethod
  def model_path(cls, model_resource_folder: str) -> str:
    return os.path.join(model_resource_folder, MyModel.MODEL_FILENAME)
