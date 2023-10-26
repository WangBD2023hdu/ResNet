import torch
from torch.utils.data import Dataset
import pickle
from PIL import Image
import os
class MyData(Dataset):

    def __init__(self, image_dir, label_dir, transform = None):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_path = os.listdir(self.image_dir)
        self.transform = transform
        self.labels = {}
        with open("data\map.pkl",'rb') as f:
            self.map = pickle.load(f)

        labels_ = open(os.path.join(self.label_dir))
        next(labels_)
        for label in labels_.readlines():
            self.labels[label.split(',')[0]] = label.split(',')[1].split('n')[1].split('\n')[0]

        # labels_ = open(os.path.join(self.label_dir, 'test.csv'))
        # next(labels_)
        # for label in labels_.readlines():
        #     self.labels[label.split(',')[0]] = label.split(',')[1].split('n')[1].split('\n')[0]
        #
        # labels_ = open(os.path.join(self.label_dir, 'val.csv'))
        # next(labels_)
        # for label in labels_.readlines():
        #     self.labels[label.split(',')[0]] = label.split(',')[1].split('n')[1].split('\n')[0]
        return
    def __getitem__(self,index):
        lab_tensor = torch.zeros(100)
        img_name = self.img_path[index]
        label = self.labels[img_name]
        img = Image.open(os.path.join(self.image_dir, img_name ))
        if self.transform is not None:
            img = self.transform(img)
        lab_tensor[self.map['n'+label]-1:self.map['n'+label]] = 1
        return img, lab_tensor

    def __len__(self):

        return len(self.img_path)