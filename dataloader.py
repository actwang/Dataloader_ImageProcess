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
from sklearn.externals import joblib
from PIL import Image
from pathlib import Path

# Image size required by your neural network
HEIGHT = 256
WIDTH = 256


class DataLoader(object):

    def __init__(self, pathToImage, label, batch_size=4):
        # reading data list
        self.list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImage, '*.tif'))]
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

    def get_batch(self, state):
        # once we reach the end of the dataset, shuffle it again and reset cursor
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)
        # initialize the image tensor with arrays full of zeros
        imgs = torch.zeros(self.batch_size, 1, HEIGHT, WIDTH)
        # initialize the label tensor with zeros, 3 here is the size of one-hot encoded label for a 3-class classification problem
        labels = torch.zeros(self.batch_size, 4)

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
            imgs = image.data_transforms[state]

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
            # index() will work only in this case because the keys are also the index in the dictionary
            lab_ind = list(temp_dict.values()).index(self.label)
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


# visualize your results
# save to directories
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

# Qu: where does this go?
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
# join data_dir and x = 'train' and 'val' for directory as input along with the transform methods.
# pass them to ImageFolder and so it's a dictionary with keys 'train' and 'val'

for y in patch_lab:
    # Qu: ImageFolder just dataloader? just used DataLoader for the whole thing

    # image_datasets = {x: datasets.ImageFolder(os.path.join(patch_dir, x, y), data_transforms[x])
    #                   for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(os.path.join(patch_dir, x, y), y, batch_size=4)
               for x in ['train', 'val', 'test']}

    # don't need class names
    dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val', 'test']}
    # class_names = image_datasets['train'].classes

print(str(torch.cuda.is_available()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # changing best_model_wts doesn't change model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  # format substitutes inputs into {}
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val','test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# resnet model (followed tutorial)
# Qu: where get_batch()?
model_or = models.resnet18(pretrained=False)
num_ftrs = model_or.fc.in_features
model_or.fc = nn.Linear(num_ftrs, 4)
model_or = model_or.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_or = optim.SGD(model_or.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_or, step_size=7, gamma=0.1)

model_or = train_model(model_or, criterion, optimizer_or, exp_lr_scheduler, num_epochs=25)
