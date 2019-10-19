import glob
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
import argparse

img_dir = '/home/zi29/Desktop/IMP/wk3/assignment3/img_sets'
cursor = 0
ref_img = {}    # should become {name: 2_600.tif, name: 3_600.tif} values being images
resize_imgs = {}  # will enumerate to {name: 2_100.tif, name: 2_200.tif, name:2_300.tif, name: 3_100.tif,
                        # name:3_200.tif, name:3_300.tif}

# list of image names without .tif
current_name = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(img_dir, '*.tif'))]
# for loop to generate two dictionaries above
for idx in range(len(current_name)):
    current_dpi = int(current_name[idx][2:])
    full_img_path = os.path.join(img_dir, current_name[idx] + '.tif')
    img = cv2.imread(full_img_path, 0)

    # if it's reference image, add it to ref_img
    if current_dpi == 600:
        ref_img[current_name[idx]] = img

    # otherwise resize it and add it to resize_imgs
    else:
        resized_img = cv2.resize(img, None, fx=600 / current_dpi, fy=600 / current_dpi,
                                 interpolation=cv2.INTER_LINEAR)
        resize_imgs[current_name[idx]] = resized_img


for name, img in resize_imgs.items():
    # full image path with diff in file name
    full_img_path = os.path.join(img_dir, name + 'diff.tif')
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(ref_img[name[0] + '_600'], img, full=True)
    diff = (diff * 255).astype("uint8")  # diff is a difference image
    print("SSIM: {}".format(score))

    # visualize the difference image and save it if it makes sense
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    diff = cv2.resize(diff, (1000, 1000))
    cv2.imshow(name, diff)
    key = cv2.waitKey(0)
    if key == ord('s'):
        plt.imsave(full_img_path, diff)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
