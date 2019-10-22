import os
import cv2

imgpath = '/home/zi29/Desktop/IMP/wk4/dataset/raw/orig_imgs/train/dpi75/ENG1_1.tif'
uncrop = cv2.imread(imgpath, 0)
img = uncrop[0:256,0:256]
cv2.imshow('img',img)
cv2.waitKey(0)
#print(img,'img')
thresh, th1 = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(th1,'thresholded')
sum = 0
print(thresh)
ct =0
tot1 = 0
for lst in th1:
    for pix in lst:
        sum += pix
        ct += 1
        if pix != 0:
            tot1 += 1
avg = sum/ct
print(round(avg,2),'avg')
print(ct, 'ct')
print(tot1,'tot1')