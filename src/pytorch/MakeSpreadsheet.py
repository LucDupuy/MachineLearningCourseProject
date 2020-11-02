"""
1 = enemy
0 = no enemy
"""
import os
import pandas as pd


enemy_path = r'C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/dataset/enemy/'
none_path = r'C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/dataset/none/'
col1_list = []
col2_list = []

for r, d, f in os.walk(enemy_path):
    for file in f:
        if '.png' in file:
            col1_list.append(file)
            col2_list.append(1)

for r, d, f in os.walk(none_path):
    for file in f:
        if '.png' in file:
            col1_list.append(file)
            col2_list.append(0)

enemy_df = pd.DataFrame({'path': col1_list, 'class': col2_list})
enemy_df.to_excel('C:/Users/Luc/Documents/CPS 803/Main Project/src/pytorch/data/dataset/dataset.xlsx', header=False, index=False)