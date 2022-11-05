import json
import os
import uuid

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from nervaluate import Evaluator
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, set_seed
from argparse import ArgumentParser


class GCNNCell(nn.Module):
    def __init__(self, in_size, out_size, dropout) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_size, out_size, kernel_size=3, padding=1, bias=False)
        self.gate = nn.Conv1d(in_size, out_size, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        c = self.conv(x)
        g = self.gate(x)
        g = self.sigmoid(g)
        x = c * g
        x = self.dropout(x)
        return x


class GCNN(nn.Module):
    def __init__(self, in_size, hidden_size, output_size, num_layers, dropout) -> None:
        super().__init__()
        self.stem = GCNNCell(in_size, hidden_size, dropout)
        self.backbone = nn.Sequential(
            *[GCNNCell(hidden_size, hidden_size, dropout) for _ in range(num_layers)]
        )
        self.head = GCNNCell(hidden_size, output_size, 0)

    def forward(self, x):
        # need (N C L) have (N L C)

        x = torch.permute(x, (0, 2, 1))

        x = self.stem(x)
        x = self.backbone(x)
        x = self.head(x)

        x = torch.permute(x, (0, 2, 1)).contiguous()

        return x


class LSTMWrap(nn.Module):
    def __init__(self, lstm, lin) -> None:
        super().__init__()
        self.lstm = lstm
        self.lin = lin

    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.lin(x)
        return x


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        tokenizer,
        lr,
        lr_factor,
        lr_patience,
        optimizer,
        model_max_length,
        bio2tags,
        tag_list,
        train_head_only,
        lstm_head,
        frozen_bert_layers,
        emb_and_lstm,
        lstm_dropout,
        lstm_layers,
        lstm_size,
        emb_and_gcnn,
        gcnn_dropout,
        gcnn_layers,
        gcnn_size,
        emb_and_lstm_and_gcnn,
    ):
        super().__init__()
        self.optimizer_name = optimizer

        print("Loading AutoModel [{}] ...".format(model_name))
        self.tokenizer = tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(bio2tags), from_flax=False
        )
        if train_head_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.classifier.parameters():
                p.requires_grad = True

        if lstm_head:
            self.model.classifier = LSTMWrap(
                nn.LSTM(
                    input_size=self.model.classifier.weight.shape[1],
                    hidden_size=lstm_size,
                    num_layers=lstm_layers,
                    batch_first=True,
                    bidirectional=True,
                    proj_size=lstm_size // 2,
                    dropout=lstm_dropout,
                ),
                nn.Linear(lstm_size, len(bio2tags)),
            )

        if frozen_bert_layers:
            for p in self.model.bert.embeddings.parameters():
                p.requires_grad = False
            for i in range(frozen_bert_layers):
                layer = self.model.bert.encoder.layer[i]
                for p in layer.parameters():
                    p.requires_grad = False

        self.is_transformer = True

        if emb_and_lstm:
            self.is_transformer = False
            self.embeddings = self.model.bert.embeddings
            self.model = LSTMWrap(
                nn.LSTM(
                    input_size=self.model.classifier.weight.shape[1],
                    hidden_size=lstm_size,
                    num_layers=lstm_layers,
                    batch_first=True,
                    bidirectional=True,
                    proj_size=lstm_size // 2,
                    dropout=lstm_dropout,
                ),
                nn.Linear(lstm_size, len(bio2tags)),
            )
            self.loss = nn.CrossEntropyLoss()

        if emb_and_gcnn:
            self.is_transformer = False
            self.embeddings = self.model.bert.embeddings
            self.model = GCNN(
                self.model.classifier.weight.shape[1],
                gcnn_size,
                len(bio2tags),
                gcnn_layers,
                gcnn_dropout,
            )
            self.loss = nn.CrossEntropyLoss()

        if emb_and_lstm_and_gcnn:
            self.is_transformer = False
            self.embeddings = self.model.bert.embeddings
            self.model = nn.Sequential(
                LSTMWrap(
                    nn.LSTM(
                        input_size=self.model.classifier.weight.shape[1],
                        hidden_size=lstm_size,
                        num_layers=lstm_layers,
                        batch_first=True,
                        bidirectional=True,
                        proj_size=lstm_size // 2,
                        dropout=lstm_dropout,
                    ),
                    nn.Linear(lstm_size, gcnn_size),
                ),
                GCNN(gcnn_size, gcnn_size, len(bio2tags), gcnn_layers, gcnn_dropout),
            )
            self.loss = nn.CrossEntropyLoss()

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.model_max_length = model_max_length
        self.bio2tags = bio2tags
        self.tag_list = tag_list
        self.num_labels = len(bio2tags)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        if self.is_transformer:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss, logits = output["loss"], output["logits"]
        else:
            x = self.embeddings(input_ids=batch["input_ids"])
            logits = self.model(x)
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        token_idx = batch["token_idx"]

        loss, logits = self(batch)

        batch_size = logits.size()[0]
        batch_pred = torch.argmax(logits.detach().cpu(), dim=-1).tolist()
        batch_gold = labels.detach().cpu().tolist()
        batch_token_idx = token_idx.detach().cpu().tolist()

        for batch_idx in range(batch_size):
            pred, gold, idx = (
                batch_pred[batch_idx],
                batch_gold[batch_idx],
                batch_token_idx[batch_idx],
            )
            y_hat, y = [], []
            for i in range(0, max(idx) + 1):
                pos = idx.index(i)
                y_hat.append(pred[pos])
                y.append(gold[pos])

        return {"loss": loss, "y": y, "y_hat": y_hat}

    def log_epoch_end(self, outputs, mode):
        odf = pd.DataFrame(outputs)

        mean_val_loss = odf["loss"].mean()
        gold, pred = [], []
        for _, row in odf.iterrows():
            gold.append([self.bio2tags[token_id] for token_id in row["y"]])
            pred.append([self.bio2tags[token_id] for token_id in row["y_hat"]])

        evaluator = Evaluator(gold, pred, tags=self.tag_list, loader="list")

        results, _ = evaluator.evaluate()
        self.log(f"{mode}/avg_loss", mean_val_loss, prog_bar=True)
        self.log(f"{mode}/ent_type", results["ent_type"]["f1"])
        self.log(f"{mode}/partial", results["partial"]["f1"])
        self.log(f"{mode}/strict", results["strict"]["f1"])
        self.log(f"{mode}/exact", results["exact"]["f1"])

    def training_epoch_end(self, outputs):
        self.log_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.log_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.log_epoch_end(outputs, "test")

    def configure_optimizers(self):
        if self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08
            )
        else:
            optimizer = torch.optim.SGD(
                [p for p in self.parameters() if p.requires_grad], lr=self.lr
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.lr_factor,
                    patience=self.lr_patience,
                    mode="max",
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "valid/strict",
                "strict": True,
                "name": "learning_rate",
            },
        }

    def predict(self, input_string):
        input_ids = self.tokenizer.encode(input_string, add_special_tokens=False)

        # run the model
        output = self.model(
            input_ids=torch.unsqueeze(torch.LongTensor(input_ids), 0), return_dict=True
        )
        logits = output["logits"]

        # extract results
        indices = torch.argmax(logits.detach().cpu(), dim=-1).squeeze(dim=0).tolist()

        output_ids = []

        for id in input_ids:
            output_ids.append(self.tokenizer.decode(id))

        output_classes = []
        for i in indices:
            output_classes.append(self.bio2tags[i])

        return output_ids, output_classes


class RoNecDataset(Dataset):
    def __init__(self, instances):
        self.instances = []

        # run check
        for instance in instances:
            ok = True
            if len(instance["ner_ids"]) != len(instance["tokens"]):
                print("Different length ner_tags found")
                ok = False
            else:
                for tag, token in zip(instance["ner_ids"], instance["tokens"]):
                    if token.strip() == "":
                        ok = False
                        print("Empty token found")
            if ok:
                self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]


class RoNecCollator(object):
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.validate_pad_token()

    def validate_pad_token(self):
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.sep_token is not None:
            print(
                f"\tNo PAD token detected, automatically assigning the SEP token as PAD."
            )
            self.tokenizer.pad_token = self.tokenizer.sep_token
            return
        if self.tokenizer.eos_token is not None:
            print(
                f"\tNo PAD token detected, automatically assigning the EOS token as PAD."
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        if self.tokenizer.bos_token is not None:
            print(
                f"\tNo PAD token detected, automatically assigning the BOS token as PAD."
            )
            self.tokenizer.pad_token = self.tokenizer.bos_token
            return
        if self.tokenizer.cls_token is not None:
            print(
                f"\tNo PAD token detected, automatically assigning the CLS token as PAD."
            )
            self.tokenizer.pad_token = self.tokenizer.cls_token
            return
        raise Exception(
            "Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required."
        )

    def __call__(self, input_batch):
        batch_input_ids, batch_labels, batch_attention, batch_token_idx = [], [], [], []
        max_len = 0

        for instance in input_batch:
            instance_ids, instance_labels, instance_attention, instance_token_idx = (
                [],
                [],
                [],
                [],
            )

            for i in range(len(instance["tokens"])):
                subids = self.tokenizer.encode(
                    instance["tokens"][i], add_special_tokens=False
                )
                sublabels = [instance["ner_ids"][i]]

                if (
                    len(subids) > 1
                ):  # we have a word split in more than 1 subids, fill appropriately
                    filler_sublabel = (
                        sublabels[0] if sublabels[0] % 2 == 0 else sublabels[0] + 1
                    )
                    sublabels.extend([filler_sublabel] * (len(subids) - 1))

                instance_ids.extend(subids)  # extend with the number of subids
                instance_labels.extend(sublabels)  # extend with the number of subtags
                instance_token_idx.extend(
                    [i] * len(subids)
                )  # extend with the id of the token

                assert len(subids) == len(
                    sublabels
                )  # check for possible errors in the dataset

            if len(instance_ids) != len(instance_labels):
                print(len(instance_ids))
                print(len(instance_labels))
                print(instance_ids)
                print(instance_labels)
            assert len(instance_ids) == len(instance_labels)

            # cut to max sequence length, if needed
            if len(instance_ids) > self.max_seq_len - 2:
                instance_ids = instance_ids[: self.max_seq_len - 2]
                instance_labels = instance_labels[: self.max_seq_len - 2]
                instance_token_idx = instance_token_idx[: self.max_seq_len - 2]

            # prepend and append special tokens, if needed
            if self.tokenizer.cls_token_id and self.tokenizer.sep_token_id:
                instance_ids = (
                    [self.tokenizer.cls_token_id]
                    + instance_ids
                    + [self.tokenizer.sep_token_id]
                )
                instance_labels = [0] + instance_labels + [0]
                instance_token_idx = [
                    -1
                ] + instance_token_idx  # no need to pad the last, will do so automatically at return
            instance_attention = [1] * len(instance_ids)

            # update max_len for later padding
            max_len = max(max_len, len(instance_ids))

            # add to batch
            batch_input_ids.append(torch.LongTensor(instance_ids))
            batch_labels.append(torch.LongTensor(instance_labels))
            batch_attention.append(torch.LongTensor(instance_attention))
            batch_token_idx.append(torch.LongTensor(instance_token_idx))

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id
                else 0,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                batch_attention, batch_first=True, padding_value=0
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                batch_labels, batch_first=True, padding_value=0
            ),
            "token_idx": torch.nn.utils.rnn.pad_sequence(
                batch_token_idx, batch_first=True, padding_value=-1
            ),
        }


class MyModel:
    def __init__(self):
        self.bio2tags = [
            "O",
            "B-PERSON",
            "I-PERSON",
            "B-ORG",
            "I-ORG",
            "B-GPE",
            "I-GPE",
            "B-LOC",
            "I-LOC",
            "B-NAT_REL_POL",
            "I-NAT_REL_POL",
            "B-EVENT",
            "I-EVENT",
            "B-LANGUAGE",
            "I-LANGUAGE",
            "B-WORK_OF_ART",
            "I-WORK_OF_ART",
            "B-DATETIME",
            "I-DATETIME",
            "B-PERIOD",
            "I-PERIOD",
            "B-MONEY",
            "I-MONEY",
            "B-QUANTITY",
            "I-QUANTITY",
            "B-NUMERIC",
            "I-NUMERIC",
            "B-ORDINAL",
            "I-ORDINAL",
            "B-FACILITY",
            "I-FACILITY",
        ]
        self.tags = sorted(
            list(set([tag[2:] if len(tag) > 2 else tag for tag in self.bio2tags]))
        )
        self.seed = 42
        self.gpus = 1
        self.batch_size = 8
        self.accumulate_grad_batches = 2
        self.max_epochs = 4
        self.model_name = "dumitrescustefan/bert-base-romanian-cased-v1"
        self.tokenizer_name = "dumitrescustefan/bert-base-romanian-cased-v1"
        self.lr = 2e-05
        self.lr_factor = 2 / 3
        self.lr_patience = 5
        self.optimizer = "AdamW"
        self.gradient_clip_val = 1.0
        self.model_max_length = 512
        self.csc_train_fold = 0
        self.train_head_only = 0
        self.lstm_head = 0
        self.frozen_bert_layers = 0
        self.emb_and_lstm = 0
        self.lstm_dropout = 0
        self.lstm_layers = 4
        self.lstm_size = 512
        self.emb_and_gcnn = 0
        self.gcnn_dropout = 0.2
        self.gcnn_layers = 4
        self.gcnn_size = 512
        self.emb_and_lstm_and_gcnn = 0
        self.emb_ckpt_file = ""
        self.extra_train_file = os.path.join(os.path.dirname(__file__), "dataset-v11-38k.json")

        assert self.optimizer in ["AdamW", "SGD"]

        pl.seed_everything(self.seed, workers=True)
        set_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # init tokenizer and start loading data
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, strip_accents=False
        )
        
        self.model = TransformerModel(
            model_name=self.model_name,
            lr=self.lr,
            lr_factor=self.lr_factor,
            lr_patience=self.lr_patience,
            optimizer=self.optimizer,
            model_max_length=self.model_max_length,
            bio2tags=self.bio2tags,
            tokenizer=self.tokenizer,
            tag_list=self.tags,
            train_head_only=self.train_head_only,
            lstm_head=self.lstm_head,
            frozen_bert_layers=self.frozen_bert_layers,
            emb_and_lstm=self.emb_and_lstm,
            lstm_dropout=self.lstm_dropout,
            lstm_layers=self.lstm_layers,
            lstm_size=self.lstm_size,
            emb_and_gcnn=self.emb_and_gcnn,
            gcnn_dropout=self.gcnn_dropout,
            gcnn_layers=self.gcnn_layers,
            gcnn_size=self.gcnn_size,
            emb_and_lstm_and_gcnn=self.emb_and_lstm_and_gcnn,
        )

        self.collator = RoNecCollator(
            tokenizer=self.tokenizer, max_seq_len=self.model_max_length
        )

    def load(self, model_resource_folder):
        self.model = TransformerModel.load_from_checkpoint(
            os.path.join(model_resource_folder, "last.ckpt"),
            model_name=self.model_name,
            lr=self.lr,
            lr_factor=self.lr_factor,
            lr_patience=self.lr_patience,
            optimizer=self.optimizer,
            model_max_length=self.model_max_length,
            bio2tags=self.bio2tags,
            tokenizer=self.tokenizer,
            tag_list=self.tags,
            train_head_only=self.train_head_only,
            lstm_head=self.lstm_head,
            frozen_bert_layers=self.frozen_bert_layers,
            emb_and_lstm=self.emb_and_lstm,
            lstm_dropout=self.lstm_dropout,
            lstm_layers=self.lstm_layers,
            lstm_size=self.lstm_size,
            emb_and_gcnn=self.emb_and_gcnn,
            gcnn_dropout=self.gcnn_dropout,
            gcnn_layers=self.gcnn_layers,
            gcnn_size=self.gcnn_size,
            emb_and_lstm_and_gcnn=self.emb_and_lstm_and_gcnn,
        )

    def train(self, train_data, validation_data, model_resource_folder):
        if self.extra_train_file:
            with open(self.extra_train_file, "r", encoding="utf8") as f:
                extra_train_data = json.load(f)

        print("Loading data...")
        train_dataset = RoNecDataset(
            train_data + extra_train_data if self.extra_train_file else train_data
        )
        val_dataset = RoNecDataset(validation_data)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=self.collator,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=self.collator,
            pin_memory=True,
        )

        print("Train dataset has {} instances.".format(len(train_dataset)))
        print("Valid dataset has {} instances.".format(len(val_dataset)))
        print("Test dataset has {} instances.\n".format(len(test_dataset)))

        if self.emb_ckpt_file and hasattr(self.model, "embeddings"):
            ckpt = torch.load(self.emb_ckpt_file)
            self.model.embeddings.load_state_dict(
                {
                    k.replace("model.bert.embeddings.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if k.startswith("model.bert.embeddings")
                }
            )

        model_checkpointer = pl.callbacks.ModelCheckpoint(
            dirpath=model_resource_folder, filename="last.ckpt", save_last=True
        )

        trainer = pl.Trainer(
            gpus=self.gpus,
            max_epochs=self.max_epochs,
            callbacks=[model_checkpointer],
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=1.0,
        )
        trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, test_data):
        test_dataset = RoNecDataset(test_data)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=self.collator,
            pin_memory=True,
        )

        trainer = pl.Trainer(
            gpus=self.gpus,
        )
        
        res = trainer.test(self.model, test_dataloader)

        return res[0]['test/strict']


if __name__ == "__main__":
    with open("/home/oodapow/code/ronec/data/train.json", "r", encoding="utf8") as f:
        train_dataset = json.load(f)
    with open("/home/oodapow/code/ronec/data/valid.json", "r", encoding="utf8") as f:
        valid_dataset = json.load(f)
    with open("/home/oodapow/code/ronec/data/test.json", "r", encoding="utf8") as f:
        test_dataset = json.load(f)

    # TRAINING
    model = MyModel()
    model.train(train_dataset, valid_dataset, "output")

    # INFERENCE
    from time import perf_counter

    # load model
    model = MyModel()
    model.load("output")

    # inference
    start_time = perf_counter()
    f1_strict_score = model.predict(test_dataset)
    stop_time = perf_counter()

    print(f"Predicted in {stop_time-start_time:.2f}.")
    print(f"F1-strict score = {f1_strict_score:.5f}")  # this is the score we want :)
