import cv2
import os,sys

# Opens and resizes the image using opencv
def openAndResize(path,size):
    try:
        im = cv2.imread(path, 0)
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = size - new_size[1]
        delta_h = size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        return 1,new_im
    except:
        print(sys.exc_info()[0])
        print("Error Image : ",path)
        return -1,[] #error