#!pip install portalocker

import torchtext

xlmr_base = torchtext.models.XLMR_BASE_ENCODER

#write your code here
classifier_head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim=768)

# Tip: you can call a method from xlmr_base to load the model with the head
# write your code here
model = xlmr_base.get_model(head=classifier_head)
model

from torchtext.datasets import SST2

datapipes = {}
# write your code here
datapipes['train'] = SST2(split='train')
datapipes['val'] = SST2(split='dev')

#output for sanity
row = next(iter(datapipes['train']))
text, label = row
text, label

#retrieve transformation function from XLM-RoBERTa/write function
transform_fn = torchtext.models.XLMR_BASE_ENCODER.transform()
transform_fn(text)

def apply_transform(row):
    text, label = row
    # Use the transform_fn you retrieved in the previous cell to
    # preprocess the text
    # write your code here
    return (transform_fn(text), label)

#apply function to data point
apply_transform(row)

#sample visualization
batched_datapipe = datapipes['train'].map(apply_transform).batch(4)
batch_of_tuples = next(iter(batched_datapipe))
batch_of_tuples

#function to take batch of data points, pads sequences, converts labels to tensor
import torch
from torchtext.functional import to_tensor

padding_idx = transform_fn[1].vocab.lookup_indices(['<pad>'])[0]

def tensor_batch(batch):
    tokens = batch['token_ids']
    labels = batch['labels']
    # write your code here
    return (to_tensor(tokens, padding_value=padding_idx), torch.tensor(labels))

#apply preprocessing
for k in datapipes.keys():
    datapipes[k] = datapipes[k].map(apply_transform)
    datapipes[k] = datapipes[k].batch(16)
    datapipes[k] = datapipes[k].rows2columnar(['token_ids', 'labels'])
    datapipes[k] = datapipes[k].map(tensor_batch)

#tuple
dp_out = next(iter(datapipes['train']))
dp_out

#create dataloader
from torch.utils.data import DataLoader

dataloaders = {}
dataloaders['train'] = DataLoader(dataset=datapipes['train'], batch_size=None, shuffle=True)
dataloaders['val'] = DataLoader(dataset=datapipes['val'], batch_size=None)

#fetch small batch/visualize
dl_out = next(iter(dataloaders['train']))
dl_out

#loss function
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

#optimizer

import torch.optim as optim

#suggested learning rate
lr = 1e-5

optimizer = optim.AdamW(model.parameters(), lr=lr)

#training loop

# load tensorboard wherever you are (%load_ext tensorboard, %tensorboard --logdir)

#create instance of SummaryWriter
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/roberta')

#training loop + losses
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

batch_losses = []

## Training
for i, (batch_features, batch_targets) in tqdm(enumerate(datapipes['train'])):
    # Set the model's mode
    model.train()

    # Send batch features and targets to the device
    batch_features = batch_features.to(device)
    batch_targets = batch_targets.to(device)

    # Step 1 - forward pass
    predictions = model(batch_features)

    # Step 2 - computing the loss
    loss = loss_fn(predictions, batch_targets)

    # Step 3 - computing the gradients
    loss.backward()

    batch_losses.append(loss.item())

    writer.add_scalars(main_tag='loss',
                       tag_scalar_dict={'training': loss.item()},
                       global_step=i)

    # Step 4 - updating parameters and zeroing gradients
    optimizer.step()
    optimizer.zero_grad()

writer.close()

## Validation
with torch.inference_mode():
    val_losses = []

    for i, (val_features, val_targets) in enumerate(dataloaders['val']):
        # Set the model's mode
        model.eval()

        # Send batch features and targets to the device
        val_features = val_features.to(device)
        val_targets = val_targets.to(device)

        # Step 1 - forward pass
        predictions = model(val_features)

        # Step 2 - computing the loss
        loss = loss_fn(predictions, val_targets)

        val_losses.append(loss.item())

#inference
def predict(sequence, model, transforms_fn, categories):
    # Build a tensor of token ids out of the input sequence
    token_ids = to_tensor(transforms_fn(sequence))

    # Set the model to the appropriate mode
    model.eval()

    device = next(iter(model.parameters())).device

    # Use the model to make predictions/logits
    pred = model(token_ids.unsqueeze(0).to(device))

    # Compute the probabilities corresponding to the logits and return the top value and index
    probabilities = torch.nn.functional.softmax(pred[0], dim=0)
    values, indices = torch.topk(probabilities, 1)

    return [{'label': categories[i], 'value': v.item()} for i, v in zip(indices, values)]

#try prediction function and fine-tuned model
categories = ['negative', 'positive']
text = "I really enjoy data science and WiDS!"
predict(text, model, xlmr_base.transform(), categories)

text = "2024 is way too hard!"
predict(text, model, xlmr_base.transform(), categories)