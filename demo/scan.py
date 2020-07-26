import cv2
import numpy as np
from cv2 import imwrite,imshow,waitKey,boundingRect, countNonZero, cvtColor, drawContours, findContours, getStructuringElement, imread, morphologyEx, pyrDown, rectangle, threshold

large = imread('sample/test.jpg')
# downsample and use it for processing
rgb = pyrDown(large)
# apply grayscale
small = cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# morphological gradient
morph_kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = morphologyEx(small, cv2.MORPH_GRADIENT, morph_kernel)
# binarize
_, bw = threshold(src=grad, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
morph_kernel = getStructuringElement(cv2.MORPH_RECT, (9, 1))
# connect horizontally oriented regions
connected = morphologyEx(bw, cv2.MORPH_CLOSE, morph_kernel)
mask = np.zeros(bw.shape, np.uint8)
# find contours
contours, hierarchy = findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

listRect = []
# filter contours
for idx in range(0, len(hierarchy[0])):
    rect = x, y, rect_width, rect_height = boundingRect(contours[idx])
    # fill the contour
    mask = drawContours(mask, contours, idx, (255, 255, 2555), cv2.FILLED)
    # ratio of non-zero pixels in the filled region
    r = float(countNonZero(mask)) / (rect_width * rect_height)
    print(r)
    if r > 0.45 and r < 10 and rect_height > 8 and rect_width > 8:
        listRect.append(rect)
        # rgb = rectangle(rgb, (x, y+rect_height), (x+rect_width, y), (0,255,0),3)


bwimg = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
bwimg = cv2.medianBlur(bwimg,7)
denoised = cv2.fastNlMeansDenoising(bwimg,None,3.0,21,7)
# thresholded image
threshimg = cv2.adaptiveThreshold(denoised,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)

# Simple man's smart logic to only detect letters
# - If length > 4 * height not a letter
# - If inside another rect ignore
#  rect = [x, y, rect_width, rect_height]
filteredList = []
for rect in listRect:
    filteredList.append(rect)
    # if not(rect[2] >= 4*rect[3] or rect[3] >= 4*rect[2]):
            # filteredList.append(rect)


for rect in filteredList:
    x, y, rect_width, rect_height = rect
    threshimg = rectangle(threshimg, (x, y+rect_height), (x+rect_width, y), (255,255,255),3)

imshow('',threshimg)
waitKey(0)