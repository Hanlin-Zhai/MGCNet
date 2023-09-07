import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset


class ToTensor(object):
    def __call__(self, sample):
        entry = {}
        for k in sample:
            if k == 'rect':
                entry[k] = torch.IntTensor(sample[k])
            else:
                entry[k] = torch.FloatTensor(sample[k])
        return entry


class SSTDataset(Dataset):
    def __init__(self, root_dir='', im_size=(256, 256), transform=None):
        self.filenames = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform
        self.im_size = im_size
        np.random.seed(2023)

    def __len__(self):
        return len(self.filenames)

    def read_image(self, filepath):
        image = cv2.imread(filepath)
        im_scaled = np.transpose(image, [2, 0, 1])
        return im_scaled

    def __getitem__(self, idx):
        image = self.read_image(os.path.join(self.root_dir, self.filenames[idx]))
        sample = {'gt': image}
        if self.transform:
            sample = self.transform(sample)
        return sample
