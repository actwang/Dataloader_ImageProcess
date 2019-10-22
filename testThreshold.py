import os
import cv2

imgpath = '/home/zi29/Desktop/IMP/wk4/dataset/raw/orig_imgs/train/dpi75/ENG1_1.tif'
uncrop = cv2.imread(imgpath, 0)
# for quick adjust crop area to test different patches of an image
x = 300
img = uncrop[x:x+256,x:x+256]
cv2.imshow('img',img)
cv2.waitKey(0)
# print(img,'img')

thresh, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(th1, 'thresholded')
print(thresh)

sum = 0
ct = 0
tot1 = 0    # total number of 1's
for lst in th1:
    for pix in lst:
        sum += pix
        ct += 1
        if pix != 0:
            tot1 += 1
avg = sum/ct

# we can tell from here, it's better to just stick with judging text/blank with
# threshold retval(thresh in this case) and not use average or count total number of 1's
print(round(avg,2),'avg')
print(ct, 'ct')
print(tot1,'tot1')