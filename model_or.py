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
import torchvision
from torchvision import datasets, models, transforms
from .LeNet import LeNet5
from .dataloader import DataLoader

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


# EDIT GET_BATCH()
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # changing best_model_wts doesn't change model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  # format substitutes inputs into {}
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in DataLoader.get_batch(model, phase):
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


patch_dir = '/home/zi29/Desktop/IMP/wk4/dataset/raw/patches'
patch_set = ['train', 'test', 'val']
patch_lab = ['dpi75', 'dpi150', 'dpi300', 'dpi600']

# trying to get train: all train images and val: all val images
dataloaders = {x: DataLoader(os.path.join(patch_dir, x), batch_size=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
print(str(torch.cuda.is_available()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LeNet take input 128x128
model_or = LeNet5()
model_or = model_or.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_or = optim.SGD(model_or.parameters(), lr=0.1, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_or, step_size=7, gamma=0.1)

model_or = train_model(model_or, criterion, optimizer_or, exp_lr_scheduler, num_epochs=25)
