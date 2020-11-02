'''
This file takes the images and puts them into a spreadsheet for easier use
'''

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class ImportDataset(Dataset):
    def __init__(self, excel_file, dir, transform=None):
        self.dataset = pd.read_excel(excel_file)
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)  # Number of images

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.dataset.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.dataset.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return image, y_label
