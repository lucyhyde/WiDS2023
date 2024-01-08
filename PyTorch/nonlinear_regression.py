import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['mpg', 'cyl', 'disp', 'hp', 'weight', 'acc', 'year', 'origin']

df = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
print(df)

# shuffle dataset and test/train/validation split
from sklearn.model_selection import train_test_split

shuffle = df.sample(frac=1, random_state=1).reset_index(drop=True)

raw_data = {}
trainval, raw_data['test'] = train_test_split(shuffle, test_size=0.16, shuffle=False)
raw_data['train'], raw_data['val'] = train_test_split(trainval, test_size=0.2, shuffle=False)

# clean-up
for k in raw_data.keys():
    raw_data[k].dropna(inplace=True)

import torch
from sklearn.preprocessing import StandardScaler

# create/train standardscaler, return pytorch tensor w/ standardized features
def standardize(df, contr_attr, scaler=None)
    cont_X = df[contr_attr].values
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(cont_X)
    cont_X = scaler.transform((cont_X))
    cont_X = torch.as_tensor(cont_X, dtype=torch.float32)

    return cont_X, scaler

# standardize all continuous attributes - train scaler on validation/test sets
cont_attr = ['disp', 'hp', 'weight', 'acc']

cont_data = {'train': None, 'val': None, 'test': None}

cont_data['train'], scaler = standardize(raw_data['train'], cont_attr)
cont_data['val'], _ = standardize(raw_data['val'], cont_attr, scaler)
cont_data['test'], _ = standardize(raw_data['test'], cont_attr, scaler)

from sklearn.preprocessing import OrdinalEncoder
def encode(df, cat_attr, encoder=None):
    # write your code here
    cat_X = df[cat_attr].values
    if encoder is None:
        encoder = OrdinalEncoder()
        encoder.fit(cat_X)
    cat_X = encoder.transform(cat_X)
    cat_X = torch.as_tensor(cat_X, dtype=torch.int)

    return cat_X, encoder

disc_attr = ['cyl', 'origin']

cat_data = {'train': None, 'val': None, 'test': None}
# write your code here
cat_data['train'], encoder = encode(raw_data['train'], disc_attr)
cat_data['val'], _ = encode(raw_data['val'], disc_attr, encoder)
cat_data['test'], _ = encode(raw_data['test'], disc_attr, encoder)

encoder.categories_

cat_data['train'][:, 0].unique(), cat_data['train'][:, 1].unique()

target_data = {'train': None, 'val': None, 'test': None}
target_col = 'mpg'

for k in raw_data.keys():
    target_data[k] = torch.as_tensor(raw_data[k][[target_col]].values, dtype=torch.float32)

from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, raw_data, cont_attr, disc_attr, target_col, scaler=None, encoder=None):
        # write your code here
        self.n = raw_data.shape[0]
        self.target = torch.as_tensor(raw_data[[target_col]].values, dtype=torch.float32)
        self.cont_data, self.scaler = standardize(raw_data, cont_attr, scaler)
        self.cat_data, self.encoder = encode(raw_data, disc_attr, encoder)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # write your code here
        features = (self.cont_data[idx], self.cat_data[idx])
        target = self.target[idx]

        return (features, target)

datasets = {'train': None, 'val': None, 'test': None}
# write your code here
datasets['train'] = TabularDataset(raw_data['train'], cont_attr, disc_attr, target_col)
datasets['val'] = TabularDataset(raw_data['val'], cont_attr, disc_attr, target_col, datasets['train'].scaler, datasets['train'].encoder)
datasets['test'] = TabularDataset(raw_data['test'], cont_attr, disc_attr, target_col, datasets['train'].scaler, datasets['train'].encoder)

datasets['train'][:5]

from torch.utils.data import DataLoader

dataloaders = {'train': None, 'val': None, 'test': None}
# write your code here
dataloaders['train'] = DataLoader(datasets['train'], batch_size=32, shuffle=True, drop_last=True)
dataloaders['val'] = DataLoader(datasets['val'], batch_size=16, drop_last=True)
dataloaders['test'] = DataLoader(datasets['test'], batch_size=16, drop_last=True)