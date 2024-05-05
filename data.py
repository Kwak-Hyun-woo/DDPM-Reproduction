import glob
import os
import PIL
import torch
from torchvision import transforms as T
from torchvision import datasets
from torchvision.io import read_image
import json

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.notebook import tqdm

# parameters
dataroot = "./data/celeba/"
img_dir = dataroot + "image/"
annotation = dataroot + "anno/"
image_size = 64
channels = 3
batch_size = 128

# image transformations
transforms = T.Compose([
    T.ToTensor(),
    T.Resize((image_size, image_size)),
    T.RandomHorizontalFlip(),
    # T.Normalize((0.5), (0.5))# T.Normalize([0.5],[0.5])
])

class Celeba(Dataset):
    def __init__(self, img_dir, annotation_folder = None, transforms=None, mode="train"):
        if annotation_folder is None:
            self.annotation_folder = None
        else:       
            self.annotation_folder = annotation_folder
        
        
        self.mode = mode            # 162770
        self.train_test_split_num = 162770
        self.img_num = 182638
        
        self.img_dir = img_dir
        self.transform = transforms
        self.train_image = []
        self.test_image = []
        for img in os.listdir(self.img_dir):
            if int(img.split('.')[0]) < self.train_test_split_num:
                self.train_image.append(img)
            elif int(img.split('.')[0]) >= self.train_test_split_num & int(img.split('.')[0]) < self.img_num:
                self.test_image.append(img)
        
    def __len__(self):
        if self.mode == "train":
            return len(self.train_image)
        else:
            return len(self.test_image)
                
    def __getitem__(self, idx):
        if self.mode == "train":
            if idx+1 >= self.train_test_split_num:
                raise Exception("Over Index")
        else:
            if idx+1 >= self.train_test_split_num & idx+1 < self.img_num:
                raise Exception("Over Index")
        img_path = self.img_dir + "{:0>6}.png".format(idx+1)
        image = PIL.Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image