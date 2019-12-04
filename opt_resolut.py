import cv2
import os
from Dataloader_ImageProcess.LeNet import LeNet5

# read one big image
path = "/home/zi29/Desktop/IMP/wk3/assignment3/img_sets/2_600.tif"
img = cv2.imread(path, 0)
img_shape = img.shape
height = img_shape[0]
width = img_shape[1]


def isblank(image):
    """
    takes an image and return whether or not the image is just blank without text

    :param image:
    :return: blank(bool)
    """
    retval, th1 = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 0 -> black 1 -> white
    # but thresholding value retval still range from 0 to 255, if it falls below 220,
    # most likely it contains some type of text
    return retval < 220.0


def gen_patch(image, pic_name):
    """
    takes a val/test image with its label, name and flag(test or val), crop to generate its patches,
    saving them to the s directories

    image, str(e.g. 'ENG1_1')  -> None"""

    h_steps = int(height / 128) - 1
    w_steps = int(width / 128) - 1
    y = 0
    idx = 1
    for h_move in range(h_steps):
        x = 0
        for w_move in range(w_steps):
            crop_img = image[y:y+128, x:x+128]
            x += 128
            cv2.imwrite("/home/zi29/Desktop/IMP/opt_resolute" + pic_name + '(' + ')'+str(idx) + '.tif', crop_img)
            idx += 1
        y += 128


# generate its patches
gen_patch(img, '2_600')

# create list of image patches
img_lst = []
for picname in os.listdir("/home/zi29/Desktop/IMP/opt_resolute"):
    img = cv2.imread(os.path.join("/home/zi29/Desktop/IMP/opt_resolute", picname))
    if img is not None:
        img_lst.append(img)

# LeNet take input 128x128
model = LeNet5()

# feed patches into nn
res_dict = {}
idx = 1
for pic in img_lst:
    res_dict[idx] = model(pic)
    idx += 1

# max pooling
max_res = 75
for resolution in res_dict.values():
    if resolution > max_res:
        max_res = resolution

print('oprimal resolution = ' + str(max_res))
