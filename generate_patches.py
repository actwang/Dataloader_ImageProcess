import cv2
import os
import glob

# height and width of training and val/test
HW1 = 256
HW2 = 128

# directories
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
            if not isblank(crop_img):
                cv2.imwrite('/'.join((train_sdir, label, pic_name)) + '('+str(idx)+')' + '.tif', crop_img)

            else:
                cv2.imwrite('/'.join((train_sdir, 'dpi75', pic_name)) + '('+str(idx)+')' + '.tif', crop_img)
            idx += 1
        y += 128


def valtest_gen_patch(image, label, pic_name, flag):
    """
    takes a val/test image with its label, name and flag(test or val), crop to generate its patches,
    saving them to the s directories

    image, str(e.g. 'dpi75'),str(e.g. 'ENG1_1'), str  -> None"""

    dir_dict = {'test': test_sdir, 'val': val_sdir}

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
            if not isblank(crop_img):
                cv2.imwrite('/'.join((dir_dict[flag], label, pic_name)) + '('+str(idx)+')' + '.tif', crop_img)

            else:
                cv2.imwrite('/'.join((dir_dict[flag], 'dpi75', pic_name)) + '('+str(idx)+')' + '.tif', crop_img)
            idx += 1
        y += 128


def isblank(img):
    """
    takes an image and return whether or not the image is just blank without text

    :param img:
    :return: blank(bool)
    """
    retval, th1 = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 0 -> black 1 -> white
    # but thresholding value retval still range from 0 to 255, if it falls below 220,
    # most likely it contains some type of text
    if retval < 220.0:
        blank = False
    else:
        blank = True
    return blank


# MAIN
state_dict = {'train': train_dir, 'test': test_dir, 'val': val_dir}
for state, state_dir in state_dict.items():
    # walk under train_dir into folders of different dpi's
    for root, labels, files in os.walk(state_dir):
        # labels is a list of strings
        for dpi_fdr in labels:
            img_namelst = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(state_dir, dpi_fdr, '*.tif'))]
            for name in img_namelst:
                # for rt, dirs, imgs in os.walk('/'.join((train_dir, dpi_fdr))):
                #     for img_name in imgs:
                pic = cv2.imread(os.path.join(state_dir, dpi_fdr, name+'.tif'), 0)  # read in grayscale for thresholding
                if state_dir == train_dir:
                    train_gen_patch(pic, dpi_fdr, name)
                else:
                    valtest_gen_patch(pic, dpi_fdr, name, state)
