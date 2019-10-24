import glob
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Image size required by your neural network
HEIGHT = 256
WIDTH = 256


class DataLoader(object):

    def __init__(self, pathToImage, label, batch_size = 4):
        # reading data list
        self.list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImage,'*.tif'))]
        # store the batch size
        self.batch_size = batch_size
        # store the total number of images
        self.size = len(self.list_img)
        # initialize a cursor to keep track of the current image index 
        self.cursor = 0
        # store the number of batches
        self.num_batches = self.size // batch_size
        # store image path
        self.path = pathToImage
        # label as parent directory
        self.label = label

    def get_batch(self, state, ):
        # once we reach the end of the dataset, shuffle it again and reset cursor
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)
        # initialize the image tensor with arrays full of zeros
        imgs = torch.zeros(self.batch_size, 1, HEIGHT, WIDTH)
        # initialize the label tensor with zeros, 3 here is the size of one-hot encoded label for a 3-class classification problem
        labels = torch.zeros(self.batch_size, 4)
        # compose a series of random transforms to do some runtime data augmentation
        to_tensor = transforms.Compose([
            transforms.RandomCrop(size=(HEIGHT, WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for idx in range(self.batch_size):
            # get the current file name pointed by the cursor
            curr_file = self.list_img[self.cursor]
            print('cursor' + str(self.cursor))
            # get the full path to that image
            full_img_path = os.path.join(self.path, curr_file + '.tif')
            # update cursor
            self.cursor += 1

            temp_dict = {0: 'dpi600', 1: 'dpi300', 2: 'dpi150', 3: 'dpi75'}

            # read image in grayscale
            image = cv2.imread(full_img_path, 0)
            h, w = image.shape
            if state == 'train':
                # center crop to 128x128
                imgs = image[h/2-64:h/2+64, w/2-64:w/2+64]
            else:
                imgs = image


            # # randomly resize an image to 600, 500 or 400 dpi and resize it back to 600dpi
            # temp_dict = {0: 600, 1: 500, 2: 400}
            # rand_int = np.random.randint(3)
            # if rand_int != 0:
            #     resized_img = cv2.resize(image, None, fx=temp_dict[rand_int]/600, fy=temp_dict[rand_int]/600,
            #                              interpolation=cv2.INTER_AREA)
            #     resized_img = cv2.resize(resized_img, None, fx=600/temp_dict[rand_int], fy=600/temp_dict[rand_int],
            #                              interpolation=cv2.INTER_LINEAR)
            # else:
            #     resized_img = image

            # augumentation
            # imgs[idx,0,:,:] = to_tensor(Image.fromarray(resized_img))
            #

            # label index
            lab_ind = list(temp_dict.values().index(self.label))
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
    plt.imshow(inp,cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


def showABatch(batch, title=None):
    imgs, labels = batch
    label_dict = {0: '600 dpi', 1: '300 dpi', 2: '150 dpi', 3: '75 dpi'}
    # ADD YOUR CODE HERE
    for i in range(len(batch)):
        plt.figure()
        imshow(imgs[i].squeeze(),'label: '+label_dict[torch.max(labels[i]).item()])
    plt.show()


# visualize your results
# save to directories
patch_dir = '/home/zi29/Desktop/IMP/wk4/dataset/raw/patches'
patch_set = ['train','test','val']
patch_lab = ['dpi75','dpi150','dpi300','dpi600']

for folder in patch_set:
    for label in patch_lab:
        img_path = '/'.join((patch_dir,folder,label))
        loader = DataLoader(img_path,label, batch_size=4)
        print(loader.batch_size)
        print(loader.num_batches)

# N_batch = 3
# for i in range(N_batch):
#     showABatch(trainLoader.get_batch('train'))
#     showABatch(testLoader.get_batch('test'))
#     showABatch(valLoader.get_batch('val'))
