import numpy as np
import torch
import re
from CanineForTokenClassificationCustom import CanineReviewClassifier

from datasets import load_dataset

dataset = load_dataset("dumitrescustefan/diacritic")
val_ds = dataset["validation"].select(list(range(100)))

max_length = 256

labels = [
    "no_diac", 
    "ă", 
    'î',
    "â",
    "ș",
    "ț",
    'Î',
    'Â',
    'Ă',
    'Ș',
    'Ț',
]

chars_with_diacritics = ['a','t','i','s', 'A', 'T', 'I', 'S']

id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

from transformers import CanineTokenizer

tokenizer = CanineTokenizer.from_pretrained("google/canine-s")

val_ds = val_ds.rename_column("text", "input")
val_ds = val_ds.map(lambda examples: tokenizer(examples['input'], padding="max_length", truncation=True, max_length=max_length),
                        batched=True)


from torch.utils.data import DataLoader






model = CanineReviewClassifier.load_from_checkpoint('epoch=6-step=5467.ckpt')
# model = CanineReviewClassifier()



text = "Fata sta in fata cu camasa de in in mana stanga."
# prepare text for the model
encoding = tokenizer(text, return_tensors="pt")

# forward pass
outputs = model(**encoding)



# print(tokenizer.decode(batch['input_ids'][2])[:100])
# print(batch["labels"][2])


# l = batch["labels"][2][1:100]
# i = tokenizer.decode(batch['input_ids'][2][1:])[:100]
# for idx, (c_l, c_i) in enumerate(zip(l,i)):
#     if c_l != -1 and c_l != 0:
#         print(c_l, c_i, idx)
        

classes = outputs.logits.argmax(-1)[0,1:-1]
result = ''
for i, c in enumerate(text):
    result_c = c if classes[i].item() == 0 or c not in chars_with_diacritics  else id2label[classes[i].item()]
    if i > 0 and  result[i-1] != ' ':
        if result_c == 'Î':
            result_c = 'î'
        elif result_c == 'Â':
            result_c = 'â'
        elif result_c == 'Ă':
            result_c = 'ă'
        elif result_c == 'Ș':
            result_c = 'ș'
        elif result_c == 'Ț':
            result_c = 'ț'
    result += result_c
print(text)
print(result)