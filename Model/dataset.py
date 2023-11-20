import numpy as np
from torch.utils.data import Dataset
from glob import glob
import scipy.io as sio
import logging
import torch
from pathlib import Path
from os.path import splitext
from os import listdir
from PIL import Image
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im
class TrainDataSet(Dataset):
    def __init__(self, receiver1_dir, receiver2_dir, n_class):
        self.R1_dir=receiver1_dir
        self.R2_dir=receiver2_dir
        self.receiver1_dir = Path(receiver1_dir)
        self.receiver2_dir = Path(receiver2_dir)
        self.n_class=n_class
        self.ids1 = [splitext(file)[0] for file in listdir(receiver1_dir) if not file.startswith('.')]
        if not self.ids1:
            raise RuntimeError(f'No input file found in {receiver1_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids1)} examples')
        self.ids2 = [splitext(file)[0] for file in listdir(receiver2_dir) if not file.startswith('.')]
        if not self.ids2:
            raise RuntimeError(f'No input file found in {receiver2_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids2)} examples')
        self.labels = np.zeros((1,len(self.ids2)))[0]
        for i in range(len(self.ids1)):
            temp= self.ids1[i].split('1')[0]

            if temp=="UserA":
                self.labels[i]=0
            elif temp=="UserB":
                self.labels[i] =1
            elif temp == "UserC":
                self.labels[i] =2
            elif temp == "UserD":
                self.labels[i] = 3
            elif temp == "UserE":
                self.labels[i] = 4
            elif temp == "UserF":
                self.labels[i] = 5
            elif temp == "UserI":
                self.labels[i] =6
            elif temp == "UserJ":
                self.labels[i] =7
            elif temp == "UserK":
                self.labels[i] =8
            elif temp == "UserL":
                self.labels[i] =9
            elif temp == "UserM":
                self.labels[i] = 10
            elif temp == "UserN":
                self.labels[i] = 11
            elif temp == "UserO":
                self.labels[i] = 12
            elif temp == "UserP":
                self.labels[i] = 13
            elif temp == "UserQ":
                self.labels[i] = 14
            elif temp == "UserR":
                self.labels[i] = 15
            elif temp == "UserS":
                self.labels[i] = 16
            elif temp == "UserT":
                self.labels[i] = 17
            elif temp == "UserU":
                self.labels[i] = 18
            else:
                self.labels[i] =19

    def __getitem__(self, index):
        name1 = self.ids1[index]
        name2 = self.ids2[index]
        receiver1_file = glob(self.R1_dir+'/'+name1+'.mat')
        receiver2_file = glob(self.R2_dir+'/'+name2+'.mat')

        assert len(receiver1_file) == 1, f'Either no image or multiple images found for the ID {name1}: {receiver1_file}'
        assert len(receiver2_file) == 1, f'Either no image or multiple image found for the ID {name2}: {receiver2_file}'

        receiver1_data = sio.loadmat(receiver1_file[0])
        receiver2_data = sio.loadmat(receiver2_file[0])

        label = self.labels[index]
        return [torch.tensor(np.array([receiver1_data['Data_save']])), torch.tensor(np.array([receiver2_data['Data_save']])),
                torch.tensor(label.copy()).long().contiguous()]
    def __len__(self):
        return len(self.ids1)
class TestDataSet(Dataset):
    def __init__(self, receiver1_dir, receiver2_dir):
        self.R1_dir = receiver1_dir
        self.R2_dir = receiver2_dir
        self.receiver1_dir = Path(receiver1_dir)
        self.receiver2_dir = Path(receiver2_dir)
        self.ids1 = [splitext(file)[0] for file in listdir(receiver1_dir) if not file.startswith('.')]
        if not self.ids1:
            raise RuntimeError(f'No input file found in {receiver1_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids1)} examples')
        self.ids2 = [splitext(file)[0] for file in listdir(receiver2_dir) if not file.startswith('.')]
        if not self.ids2:
            raise RuntimeError(f'No input file found in {receiver2_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids2)} examples')
        self.labels = np.ones((1, len(self.ids2)))
    def __getitem__(self, index):
        name1 = self.ids1[index]
        name2 = self.ids2[index]
        receiver1_file = glob(self.R1_dir+'/'+name1+'.mat')
        receiver2_file = glob(self.R2_dir+'/'+name2+'.mat')

        assert len(receiver1_file) == 1, f'Either no image or multiple images found for the ID {name1}: {receiver1_file}'
        assert len(receiver2_file) == 1, f'Either no image or multiple image found for the ID {name2}: {receiver2_file}'

        receiver1_data = sio.loadmat(receiver1_file[0])
        receiver2_data = sio.loadmat(receiver2_file[0])
        label = self.labels[0][index]
        return [torch.tensor(np.array([receiver1_data['Data_save']])), torch.tensor(np.array([receiver2_data['Data_save']])),
                torch.tensor(label.copy()).long().contiguous()]
    def __len__(self):
        return len(self.ids1)
