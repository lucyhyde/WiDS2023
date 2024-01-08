import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['mpg', 'cyl', 'disp', 'hp', 'weight', 'acc', 'year', 'origin']

df = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
raw_data = {}
trainval, raw_data['test'] = train_test_split(shuffled, test_size=0.16, shuffle=False)
raw_data['train'], raw_data['val'] = train_test_split(trainval, test_size=0.2, shuffle=False)

for k in raw_data.keys():
    raw_data[k].dropna(inplace=True)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

def standardize(df, cont_attr, scaler=None):
    cont_X = df[cont_attr].values
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(cont_X)
    cont_X = scaler.transform(cont_X)
    cont_X = torch.as_tensor(cont_X, dtype=torch.float32)
    return cont_X, scaler

def encode(df, cat_attr, encoder=None):
    cat_X = df[cat_attr].values
    if encoder is None:
        encoder = OrdinalEncoder()
        encoder.fit(cat_X)
    cat_X = encoder.transform(cat_X)
    cat_X = torch.as_tensor(cat_X, dtype=torch.int)
    return cat_X, encoder


from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, raw_data, cont_attr, disc_attr, target_col, scaler=None, encoder=None):
        self.n = raw_data.shape[0]
        self.target = torch.as_tensor(raw_data[[target_col]].values, dtype=torch.float32)
        self.cont_data, self.scaler = standardize(raw_data, cont_attr, scaler)
        self.cat_data, self.encoder = encode(raw_data, disc_attr, encoder)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        features = (self.cont_data[idx], self.cat_data[idx])
        target = self.target[idx]
        return (features, target)


cont_attr = ['disp', 'hp', 'weight', 'acc']
disc_attr = ['cyl', 'origin']
target_col = 'mpg'

datasets = {'train': None, 'val': None, 'test': None}
datasets['train'] = TabularDataset(raw_data['train'], cont_attr, disc_attr, target_col)
datasets['val'] = TabularDataset(raw_data['val'], cont_attr, disc_attr, target_col,
                                 datasets['train'].scaler, datasets['train'].encoder)
datasets['test'] = TabularDataset(raw_data['test'], cont_attr, disc_attr, target_col,
                                  datasets['train'].scaler, datasets['train'].encoder)

from torch.utils.data import DataLoader

dataloaders = {'train': None, 'val': None, 'test': None}
dataloaders['train'] = DataLoader(datasets['train'], batch_size=32, shuffle=True, drop_last=True)
dataloaders['val'] = DataLoader(datasets['val'], batch_size=16, drop_last=True)
dataloaders['test'] = DataLoader(datasets['test'], batch_size=16, drop_last=True)

(cont_feat, cat_feat), targets = next(iter(dataloaders['train']))

encoder = datasets['train'].encoder

embedding_layers = []

# write your code here
emb_dim = 3

for categories in encoder.categories_:
    # write your code here
    layer = nn.Embedding(len(categories), emb_dim)
    embedding_layers.append(layer)

embedding_layers

embeddings = []

for i in range(encoder.n_features_in_):
    data = cat_feat[:5, i]

    # write your code here
    emb_values = embedding_layers[i](data)

    embeddings.append(emb_values)

embeddings

torch.cat(embeddings, 1)

import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, n_cont, cat_list, emb_dim):
        super().__init__()

        # Embedding layers
        embedding_layers = []
        # Creates one embedding layer for each categorical feature
        # just like you did in the previous section
        # write your code here
        for categories in cat_list:
            embedding_layers.append(nn.Embedding(len(categories), emb_dim))

        self.emb_layers = nn.ModuleList(embedding_layers)

        # Total number of embedding dimensions
        self.n_emb = len(cat_list) * emb_dim
        self.n_cont = n_cont

        # Linear Layer(s)
        lin_layers = []

        # The input layers takes as many inputs as the number of continuous features
        # plus the total number of concatenated embeddings

        # The number of outputs is your own choice
        # Optionally, add more hidden layers, don't forget to match the dimensions if you do

        # write your code here
        lin_layers.append(nn.Linear(self.n_emb + self.n_cont, 20))

        self.lin_layers = nn.ModuleList(lin_layers)

        # The output layer must have as many inputs as there were outputs in the last hidden layer
        # write your code here
        self.output_layer = nn.Linear(self.lin_layers[-1].out_features, 1)

        # Layer initialization - initialization scheme
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data, nonlinearity='relu')
        nn.init.kaiming_normal_(self.output_layer.weight.data, nonlinearity='relu')

    def forward(self, inputs):
        # The inputs are the features as returned in the first element of a tuple
        # coming from the dataset/dataloader
        # Make sure you split it into continuous and categorical attributes according
        # to your dataset implementation of __getitem__
        cont_data, cat_data = inputs

        # Retrieve embeddings for each categorical attribute and concatenate them
        embeddings = []

        # write your code here
        for i, layer in enumerate(self.emb_layers):
            embeddings.append(layer(cat_data[:, i]))

        embeddings = torch.cat(embeddings, 1)

        # Concatenate all features together, continuous and embeddings
        # write your code here
        x = torch.cat([cont_data, embeddings], 1)

        # Run the inputs through each layer and applies an activation function to each output
        for layer in self.lin_layers:
            # write your code here
            x = layer(x)
            x = F.relu(x)

        # Run the output of the last linear layer through the output layer
        # write your code here
        x = self.output_layer(x)

        # Return the prediction
        # write your code here
        return x

scaler = datasets['train'].scaler
encoder = datasets['train'].encoder

n_cont = scaler.n_features_in_
cat_list = encoder.categories_

n_cont, cat_list

torch.manual_seed(42)

# write your code here
model = FFN(n_cont, cat_list, emb_dim=3)

# write your code here
loss_fn = nn.MSELoss()

# Suggested learning rate
lr = 1e-2

# write your code here
optimizer = optim.Adam(model.parameters(), lr=lr)

progress_bar = tqdm(range(n_epochs))

for epoch in progress_bar:
    # do your thing here

    from tqdm import tqdm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epochs = 100

    losses = torch.empty(n_epochs)
    val_losses = torch.empty(n_epochs)

    best_loss = torch.inf
    best_epoch = -1
    patience = 3

    model.to(device)

    progress_bar = tqdm(range(n_epochs))

    for epoch in progress_bar:
        batch_losses = torch.empty(len(dataloaders['train']))

        ## Training
        for i, (batch_features, batch_targets) in enumerate(dataloaders['train']):
            # Set the model to training mode
            # write your code here
            model.train()

            # Send batch features and targets to the device
            # write your code here
            batch_features[0] = batch_features[0].to(device)
            batch_features[1] = batch_features[1].to(device)
            batch_targets = batch_targets.to(device)

            # Step 1 - forward pass
            predictions = model(batch_features)

            # Step 2 - computing the loss
            loss = loss_fn(predictions, batch_targets)

            # Step 3 - computing the gradients
            # Tip: it requires a single method call to backpropagate gradients
            # write your code here
            loss.backward()

            batch_losses[i] = loss.item()

            # Step 4 - updating parameters and zeroing gradients
            # Tip: it takes two calls to optimizer's methods
            # write your code here
            optimizer.step()
            optimizer.zero_grad()

        losses[epoch] = batch_losses.mean()

        ## Validation
        with torch.inference_mode():
            batch_losses = torch.empty(len(dataloaders['val']))

            for i, (val_features, val_targets) in enumerate(dataloaders['val']):
                # Set the model to evaluation mode
                # write your code here
                model.eval()

                # Send batch features and targets to the device
                # write your code here
                val_features[0] = val_features[0].to(device)
                val_features[1] = val_features[1].to(device)
                val_targets = val_targets.to(device)

                # Step 1 - forward pass
                predictions = model(val_features)

                # Step 2 - computing the loss
                loss = loss_fn(predictions, val_targets)

                batch_losses[i] = loss.item()

            val_losses[epoch] = batch_losses.mean()

            if val_losses[epoch] < best_loss:
                best_loss = val_losses[epoch]
                best_epoch = epoch
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, 'best_model.pth')
            elif (epoch - best_epoch) > patience:
                print(f"Early stopping at epoch #{epoch}")
                break

#check the evolution of the losses
plt.plot(losses[:epoch], label='Training')
plt.plot(val_losses[:epoch], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()

#visualize a scatterplot comparing predicted and actual values of fuel consumption

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
split = 'val'
batch = list(datasets[split][:][0])
batch[0] = batch[0].to(device)
batch[1] = batch[1].to(device)
ax.scatter(datasets[split][:][1].tolist(), model(batch).tolist(), alpha=.5)
ax.plot([0, 45], [0, 45], linestyle='--', c='k', linewidth=1)
ax.set_xlabel('Actual')
ax.set_xlim([0, 45])
ax.set_ylabel('Predicted')
ax.set_ylim([0, 45])
ax.set_title('MPG')