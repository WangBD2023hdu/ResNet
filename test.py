import os
import pickle
import torch.nn.functional as F
import torch
import shutil

data = torch.zeros([12,3,5,7])
pad = [0,0,1,1]

F.pad(data, pad, mode='constant', value=2 )
data[:, :, ::2, ::2].size()

import pandas as pd
data_train = pd.read_csv(r'data\train.csv')
data_test = pd.read_csv(r'data\test.csv')
data_val = pd.read_csv(r'data\val.csv')

data_train.label.value_counts().__len__()
data_test.label.value_counts().__len__()
data_val.label.value_counts().__len__()

data = pd.concat([data_train, data_test, data_val], axis=0)
data_test = data.sample(frac=0.2)
data_test.to_csv(r"data\ormal_data\test.csv", index=0)
for i in data_test.filename:
    shutil.move(os.path.join("data", 'images', i), os.path.join("data", "test"))

data_temp = pd.concat([data,data_test], axis=0)
data_train = data_temp.drop_duplicates(subset=["filename"], keep=False)
for i in data_train.filename:
    shutil.move(os.path.join("data", "images", i), os.path.join("data", "train"))

data_train.to_csv(r"data\normal_data\train.csv", columns=0)
data_test.to_csv(r"data\normal_data\test.csv", columns=0)

data.label.value_counts().__len__()
map = {}
num=0
for i in data.label:
    if i in map.keys():
        continue
    else:
        map[i]=num
        num = num + 1
with open(r"data\map.pkl", "wb") as f:
    pickle.dump(map, f, -1)

with open(r"data\map.pkl", "rb") as f:
    t = pickle.load(f)

labels_ = open(os.path.join(r'data\normal_data', 'train.csv'))
labels = {}
next(labels_)
for label in labels_.readlines():
    print(label)
    labels[label.split(',')[0]] = label.split(',')[1].split('n')[1].split('\n')[0]

labels[label.split(',')[0]]