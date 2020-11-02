import torch
import os
from CustomDataset import ImportDataset

# This enables GPU usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#It should print out "cuda:0" if it can detect your gpu correctly
print(torch.cuda.current_device())

print(os.getcwd())


enemy_path = 'C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/dataset/enemy/0enemy0001.png'

