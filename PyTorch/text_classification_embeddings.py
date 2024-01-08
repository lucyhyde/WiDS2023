#data cleaning, getting rid of a few HTML tags, replacing some special characters etc

import numpy as np

chr_codes = np.array([
     36,   151,    38,  8220,   147,   148,   146,   225,   133,    39,  8221,  8212,   232,   149,   145,   233,
  64257,  8217,   163,   160,    91,    93,  8211,  8482,   234,    37,  8364,   153,   195,   169
])
chr_subst = {f' #{c};':chr(c) for c in chr_codes}
chr_subst.update({' amp;': '&', ' quot;': "'", ' hellip;': '...', ' nbsp;': ' ', '&lt;': '', '&gt;': '',
                  '&lt;em&gt;': '', '&lt;/em&gt;': '', '&lt;strong&gt;': '', '&lt;/strong&gt;': ''})

def replace_chars(sent):
    to_replace = [c for c in list(chr_subst.keys()) if c in sent]
    for c in to_replace:
        sent = sent.replace(c, chr_subst[c])
    return sent

def preproc_description(desc):
    desc = desc.replace('\\', ' ').strip()
    return replace_chars(desc)

#converts the label into a 0-based numeric value, and keeps only labels and clean up text.
from torchdata.datapipes.iter import FileLister
from torch.utils.data import DataLoader

def create_raw_datapipe(fname):
    datapipe = FileLister(root='.')
    datapipe = datapipe.filter(filter_fn=lambda v: v.endswith(fname))
    datapipe = datapipe.open_files(mode='rt', encoding="utf-8")
    datapipe = datapipe.parse_csv(delimiter=",", skip_lines=0)
    datapipe = datapipe.map(lambda row: (int(row[0])-1, preproc_description(row[2])))
    return datapipe

datapipes = {}
datapipes['train'] = create_raw_datapipe('train.csv').shuffle(buffer_size=125000)
datapipes['test'] = create_raw_datapipe('test.csv')

#tokenization and embedding

from torch.utils.data import DataLoader

dataloaders = {}
dataloaders['train'] = DataLoader(dataset=datapipes['train'], batch_size=4, shuffle=True)
dataloaders['test'] = DataLoader(dataset=datapipes['test'], batch_size=4)

#test visualization
labels, sentences = next(iter(dataloaders['train']))
labels, sentences

#function to return list of tokens
from torchtext.data import get_tokenizer


def tokenize_batch(sentences, tokenizer=None):
    # Create the basic tokenizer if one isn't provided
    # write your code here
    if tokenizer is None:
        tokenizer = get_tokenizer('basic_english')

    # Tokenize sentences and returns the result
    # write your code here
    return [tokenizer(s) for s in sentences]

#test visualization
tokens = tokenize_batch(sentences)
tokens

[len(s) for s in tokens]

#padding
def fixed_length(tokens_batch, max_len=128, pad_token='<pad>'):
    # Truncate every sentence to max_len
    # write your code here
    truncated = [s[:max_len] for s in tokens_batch]

    # Check the actual maximum length of the (truncated) inputs
    # write your code here
    current_max = max([len(s) for s in truncated])

    # Appends as many padding tokens as necessary to make every
    # sentence as long as the actual maximum length
    # write your code here
    padded = [s + [pad_token] * (current_max - len(s)) for s in truncated]
    return padded

#test vis
lengths = [len(s) for s in fixed_length(tokens)]
lengths

#decompress GloVe vectors
import os
from torchtext.vocab import GloVe

new_locations = {key: os.path.join('https://huggingface.co/stanfordnlp/glove/resolve/main',
                                   os.path.split(GloVe.url[key])[-1]) for key in GloVe.url.keys()}
GloVe.url = new_locations

vec = GloVe(name='6B', dim=50)

#function (list of lists of tokens, instance of Vectors, retrieves embeddings)
import torch


def get_embeddings(tokens, vec):
    # Pad all lists so they have matching lengths
    padded = fixed_length(tokens)

    # Retrieve embeddings from the Vector object using `get_vecs_by_tokens`
    embeddings = torch.cat([vec.get_vecs_by_tokens(s).unsqueeze(0) for s in padded], dim=0)

    return embeddings

#test visualization
embeddings = get_embeddings(tokens, vec)
embeddings.shape

embeddings

#bag of embeddings
embeddings = vec.get_vecs_by_tokens(tokens[0])
embeddings.shape

boe = embeddings.mean(axis=0)
boe.shape


def get_bag_of_embeddings(tokens, vec):
    # Retrieve embeddings from the Vector object using `get_vecs_by_tokens`
    # For every list of tokens, take the average of their embeddings
    # Make sure to get the shapes right, and concatenate the tensors so
    # the resulting shape is N, D
    # write your code here
    embeddings = torch.cat([vec.get_vecs_by_tokens(s).mean(axis=0).unsqueeze(0) for s in tokens], dim=0)

    return embeddings

boe = get_bag_of_embeddings(tokens, vec)
boe.shape

#datapipes
datapipes = {}
datapipes['train'] = create_raw_datapipe('train.csv').shuffle(buffer_size=125000)
datapipes['test'] = create_raw_datapipe('test.csv')

dataloaders = {}
dataloaders['train'] = DataLoader(dataset=datapipes['train'], batch_size=32, shuffle=True)
dataloaders['test'] = DataLoader(dataset=datapipes['test'], batch_size=32)

#training loop
import torch
import torch.nn as nn

torch.manual_seed(11)
model = nn.Sequential(nn.Linear(50, 4))

loss_fn = nn.CrossEntropyLoss()

import torch.optim as optim

# Suggested learning rate
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

#training loop
vec = GloVe(name='6B', dim=50)

batch_losses = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

## training
for i, batch in enumerate(dataloaders['train']):
    # Set the model's mode
    model.train()

    # Unpack your batch (it has labels and sentences) and Tokenize the sentences, and compute their bags of embeddings
    labels, sentences = batch
    tokens = tokenize_batch(sentences)
    embeddings = get_bag_of_embeddings(tokens, vec)

    embeddings = embeddings.to(device)
    labels = labels.to(device)

    # Step 1 - forward pass
    predictions = model(embeddings)

    # Step 2 - computing the loss
    loss = loss_fn(predictions, labels)

    # Step 3 - computing the gradients
    loss.backward()

    batch_losses.append(loss.item())

    # Step 4 - updating parameters and zeroing gradients
    optimizer.step()
    optimizer.zero_grad()

# plot
from matplotlib import pyplot as plt
plt.plot(batch_losses)

# evaluate
import evaluate

metric1 = evaluate.load('precision', average=None)
metric2 = evaluate.load('recall', average=None)
metric3 = evaluate.load('accuracy')

model.eval()

for batch in dataloaders['test']:
    # Unpack your batch (it has labels and sentences)
    # Tokenize the sentences, and compute their bags of embeddings
    # write your code here
    labels, sentences = batch
    tokens = tokenize_batch(sentences)
    embeddings = get_bag_of_embeddings(tokens, vec)

    embeddings = embeddings.to(device)
    labels = labels.to(device)

    # write your code here
    predictions = model(embeddings)

    # write your code here
    pred_class = predictions.argmax(dim=1)

    pred_class = pred_class.tolist()
    labels = labels.tolist()

    metric1.add_batch(references=labels, predictions=pred_class)
    metric2.add_batch(references=labels, predictions=pred_class)
    metric3.add_batch(references=labels, predictions=pred_class)

metric1.compute(average=None), metric2.compute(average=None), metric3.compute()
