from __future__ import print_function, division
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision import datasets, models, transforms
from Dataloader_ImageProcess.LeNet import LeNet5
import joblib
from PIL import Image
from pathlib import Path

# Image size required by your neural network
HEIGHT = 128
WIDTH = 128
patch_dir = '/home/zi29/Desktop/IMP/wk4/dataset/raw/patches'
patch_set = ['train', 'test', 'val']
patch_lab = ['dpi75', 'dpi150', 'dpi300', 'dpi600']
# for state in patch_set:
#     for label in patch_lab:
#         img_path = '/'.join((patch_dir, state, label))
#         loader = DataLoader(img_path, label, batch_size=4)
#         print(loader.batch_size)
#         print(loader.num_batches)

# N_batch = 3
# for i in range(N_batch):
#     showABatch(trainLoader.get_batch('train'))
#     showABatch(testLoader.get_batch('test'))
#     showABatch(valLoader.get_batch('val'))
data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor()
            ]),
        'val': transforms.Compose([
            transforms.ToTensor()
            ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ])
}


class DataLoader(object):

    def __init__(self, path_to_labels, batch_size=4):
        # reading data list
        # now it's a list of all images in that dataset(train or val)
        self.list_fullimgpath = []
        for y in patch_lab:
            temp_list = glob.glob(os.path.join(path_to_labels, y, '*.tif'))
            for item in temp_list:
                self.list_fullimgpath.append(item)
        print(self.list_fullimgpath)
        # store the batch size
        self.batch_size = batch_size
        # store the total number of images
        self.size = len(self.list_fullimgpath)
        # initialize a cursor to keep track of the current image index 
        self.cursor = 0
        # store the number of batches
        self.num_batches = self.size // batch_size
        # store image path
        self.path = path_to_labels
        # get length
        self.len = len(self.list_fullimgpath)

    def get_batch(self, state):
        # once we reach the end of the dataset, shuffle it again and reset cursor
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_fullimgpath)
        # initialize the image tensor with arrays full of zeros
        imgs = torch.zeros(self.batch_size, 1, HEIGHT, WIDTH)
        # initialize the label tensor with zeros, 3 here is the size of one-hot encoded label for a 3-class classification problem
        labels = torch.zeros(self.batch_size, 4)

        for idx in range(self.batch_size):
            # get the current file path pointed by the cursor
            curr_path = self.list_fullimgpath[self.cursor]
            print('cursor' + str(self.cursor))
            # update cursor
            self.cursor += 1

            # read image in grayscale and transform
            image = Image.open(curr_path).convert('L')
            imgs = data_transforms[state](image)

            # label index
            # Here is where we use split to find out what label it belongs to
            lab_ind = patch_lab.index(curr_path.split('/')[-2])
            labels[idx][lab_ind] = 1

        return imgs, labels


def imshow(inp, title=None):
    # imshow for a tensor.
    # inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = inp.numpy()
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


def showABatch(batch, title=None):
    imgs, labels = batch
    label_dict = {0: '600 dpi', 1: '300 dpi', 2: '150 dpi', 3: '75 dpi'}
    # ADD YOUR CODE HERE
    for i in range(len(batch)):
        plt.figure()
        imshow(imgs[i].squeeze(), 'label: '+label_dict[torch.max(labels[i]).item()])
    plt.show()
