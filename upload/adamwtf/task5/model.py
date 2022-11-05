import os

import numpy as np
import torch
import transformers
from torch import optim
from tqdm.auto import tqdm

DIAC_MAP = {'ț': 't', 'ș': 's', 'Ț': 'T', 'Ș': 'S', 'Ă': 'A', 'ă': 'a', 'Â': 'A', 'â': 'a', 'Î': 'I', 'î': 'i'}


def remove_diacritics(dataset_entry):
    for diac in DIAC_MAP:
        dataset_entry['text'] = dataset_entry['text'].replace(diac, DIAC_MAP[diac])
    return dataset_entry


def remove_diacritics_example(example):
    for diac in DIAC_MAP:
        example = example.replace(diac, DIAC_MAP[diac])
    return example


def has_diacritics(text):
    """Checks if text contains romanian diacritics.
    Text can be wither a word, or a sentence with all punctuation signs.

      text: str
      return: True if text has letters 'ț', 'ș', 'Ț', 'Ș', 'Ă', 'ă', 'Â', 'â', 'Î', 'î', False otherwise
    """
    assert type(text) == str

    for letter in text:
        if DIAC_MAP.get(letter) is not None:
            return True
    return False


def is_valid_diacritization(diac_text="", no_diac_text=""):
    """Checks if "diac_text" is a possible "diacritization" of no_diac_text
      Text can be wither a word, or a sentence with all punctuation signs.

      diac_text: str
      no_diac_text: str
      return: True if valid, False otherwise
    """
    assert type(diac_text) == type(no_diac_text) == str
    assert not has_diacritics(no_diac_text)

    if len(diac_text) != len(no_diac_text):
        return False

    for diac_letter, no_diac_letter in zip(diac_text, no_diac_text):
        if DIAC_MAP.get(diac_letter, diac_letter) != no_diac_letter:
            return False

    return True


tokenizer = transformers.T5Tokenizer.from_pretrained("google/flan-t5-small")


# ================================================================================================================

class MyModel:
    def __init__(self):
        # do here any initializations you require

        self.model = transformers.T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_epochs = 1
        self.lr_scheduler = transformers.get_scheduler(name="linear",
                                                       optimizer=self.optimizer,
                                                       num_warmup_steps=0,
                                                       num_training_steps=self.num_epochs)

    def _tokenize_function(self, examples):
        return tokenizer(examples["text"], padding="max_length", max_length=50, truncation=True, return_tensors="pt")

    def load(self, model_resource_folder):
        # we'll call this code before prediction
        # use this function to load any pretrained model and any other resource, from the given folder path
        self.model.load_state_dict(torch.load(os.path.join(model_resource_folder, "model.pth.tar")))

    # *** OPTIONAL ***
    def train(self, train_data_file, validation_data_file, model_resource_folder):
        # we'll call this function right after init
        # place here all your training code
        # at the end of training, place all required resources, trained model, etc in the given model_resource_folder
        self.model.train()

        train_dset = train_data_file.map(self._tokenize_function)
        train_dset = train_dset.with_format('torch')
        train_dl = torch.utils.data.DataLoader(train_dset, 256)

        train_no_diac = validation_data_file.map(remove_diacritics).map(self._tokenize_function)
        train_no_diac_set = train_no_diac.with_format('torch')
        train_no_diac_dl = torch.utils.data.DataLoader(train_no_diac_set, 256)

        for epoch in tqdm(range(self.num_epochs)):
            running_loss = []
            progress_bar = tqdm(train_dl)
            progress_bar.set_description(f'Epoch {epoch + 1}/{self.num_epochs}')
            for x, y in zip(progress_bar, train_no_diac_dl):
                target = x['input_ids']
                target = target.view(target.size(0), -1).to(self.device)

                data = y['input_ids']
                data = data.view(data.size(0), -1).to(self.device)

                outputs = self.model(input_ids=data, labels=target)

                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss.append(loss.item())

                progress_bar.set_postfix(loss=np.mean(running_loss))

            self.lr_scheduler.step()

        torch.save(self.model.state_dict(), os.path.join(model_resource_folder, "model.pth.tar"))

    def predict(self, input_file, output_file):
        # we'll call this function after the load()
        # use this place to run the prediction
        # the input is a file that does not contain diacritics
        # the output is a file that contains diacritics and,
        #    **is identical at character level** with the input file,
        #   excepting replaced diacritic letters
        self.model.eval()

        with open(input_file, 'r') as in_f, open(output_file, 'w') as out_f:
            for text_without_diacritics in in_f:

                text_without_diacritics = "romanian to english:" + text_without_diacritics

                input_ids = tokenizer(text_without_diacritics, padding="max_length", return_tensors="pt").input_ids

                outputs = self.model.generate(input_ids, max_length=50)

                text_with_diacritics = tokenizer.decode(outputs[0], skip_special_tokens=True)

                if not is_valid_diacritization(text_with_diacritics, text_without_diacritics):
                    print("Not valid diacritization! :(")

                out_f.write(text_with_diacritics)
                out_f.write("\n")

# ================================================================================================================


def main():
    model = MyModel()

    model.predict("input.txt", "output.txt")


if __name__ == '__main__':
    main()
