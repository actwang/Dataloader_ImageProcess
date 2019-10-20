import cv2
import os

# height and width of training and val/test
HW1 = 256
HW2 = 128

# original image directories
train_dir = '/home/zi29/Desktop/IMP/wk4/dataset/raw/orig_imgs/train'
test_dir = '/home/zi29/Desktop/IMP/wk4/dataset/raw/orig_imgs/test'
val_dir = '/home/zi29/Desktop/IMP/wk4/dataset/raw/orig_imgs/val'

# save to directory
train_sdir = '/home/zi29/Desktop/IMP/wk4/dataset/raw/patches/train'
test_sdir = '/home/zi29/Desktop/IMP/wk4/dataset/raw/patches/test'
val_sdir = '/home/zi29/Desktop/IMP/wk4/dataset/raw/patches/val'

# get a sample image to get the size of the original images
# size is tuple (6600,5100)
img_shape = cv2.imread(train_dir+'/dpi75/ENG1_1.tif', 0).shape
height = img_shape[0]
width = img_shape[1]


def train_gen_patch(image, label, pic_name):
    """
    takes a training image with its label, crop to generate its patches, saving them to the s directories

    image, str(e.g. 'dpi75'),str(e.g. 'ENG1_1') -> None"""

    h_steps = int(height / 128) - 2
    w_steps = int(width / 128) - 2
    y = 0
    idx = 1
    for h_move in range(h_steps):
        x = 0
        for w_move in range(w_steps):
            crop_img = image[y:y+HW1, x:x+HW1]
            x += 128
            # check if the cropped image contains text by using threshold
            # if it does, inherit label
            if ???????? :   # do we have to use otsu's method?
                cv2.imwrite('/'.join((train_sdir, label, pic_name)) + str(idx) + '.tif', crop_img)

            else:
                cv2.imwrite('/'.join((train_sdir, 'dpi75', pic_name)) + str(idx) + '.tif', crop_img)
            idx += 1
        y += 128


def valtest_gen_patch(image, label, pic_name, flag):
    """
    takes a val/test image with its label, name and flag(test or val), crop to generate its patches,
    saving them to the s directories

    image, str(e.g. 'dpi75'),str(e.g. 'ENG1_1'), str  -> None"""

    dir_dict = {'test':test_sdir, 'val':val_sdir}

    h_steps = int(height / 128) - 1
    w_steps = int(width / 128) - 1
    y = 0
    idx = 1
    for h_move in range(h_steps):
        x = 0
        for w_move in range(w_steps):
            crop_img = image[y:y+HW2, x:x+HW2]
            x += 128
            # check if the cropped image contains text by using threshold
            # if it does, inherit label
            if text:
                cv2.imwrite('/'.join((dir_dict[flag], label, pic_name)) + str(idx) + '.tif', crop_img)

            else:
                cv2.imwrite('/'.join((dir_dict[flag], 'dpi75', pic_name)) + str(idx) + '.tif', crop_img)
            idx += 1
        y += 128
