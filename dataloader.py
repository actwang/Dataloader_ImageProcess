import glob
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Image size required by your neural network
HEIGHT = 256
WIDTH = 256

class DataLoader(object):

    def __init__(self, pathToImage, batch_size = 4):
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

    def get_batch(self):
        # once we reach the end of the dataset, shuffle it again and reset cursor
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)
        # initialize the image tensor with arrays full of zeros
        imgs = torch.zeros(self.batch_size, 1, HEIGHT, WIDTH)
        # initialize the label tensor with zeros, 3 here is the size of one-hot encoded label for a 3-class classification problem
        labels = torch.zeros(self.batch_size, 3)
        # compose a series of random tranforms to do some runtime data augmentation
        to_tensor = transforms.Compose([
            transforms.RandomResizedCrop(size=(HEIGHT, WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for idx in range(self.batch_size):
            # get the current file name pointed by the cursor
            curr_file = self.list_img[self.cursor]
            # get the full path to that image
            full_img_path = os.path.join(self.path, curr_file + '.tif')
            # update cursor
            self.cursor += 1
            
            # ADD YOUR CODE HERE
            # read image in grayscale
            image = cv2.imread(full_img_path, 0)

            # randomly resize an image to 600, 500 or 400 dpi and resize it back to 600dpi
            temp_dict = {0: 600, 1: 500, 2: 400}
            rand_int = np.random.randint(3)
            if rand_int != 0:
                resized_img = cv2.resize(image, None, fx=temp_dict[rand_int]/600, fy=temp_dict[rand_int]/600,
                                         interpolation=cv2.INTER_AREA)
                resized_img = cv2.resize(resized_img, None, fx=600/temp_dict[rand_int], fy=600/temp_dict[rand_int],
                                         interpolation=cv2.INTER_LINEAR)
            else:
                resized_img = image

            # augumentation
            imgs = to_tensor(resized_img)

            labels[idx][rand_int] = 1

        return imgs, labels


def imshow(inp, title=None):
    # imshow for a tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


def showABatch(batch, title=None):
    imgs, labels = batch
    # ADD YOUR CODE HERE
    for i in range(length(batch)):
        cv2.imshow(labels[i], imgs[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# visualize your results
training_img_path = '/home/zi29/Desktop/IMP/Week 3/assignment3/600dpi'

trainLoader = DataLoader(training_img_path, batch_size = 4)
print(trainLoader.batch_size)
print(trainLoader.num_batches)

N_batch = 3
for i in range(N_batch):
    showABatch(trainLoader.get_batch())
