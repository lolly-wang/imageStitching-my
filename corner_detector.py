'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
from skimage.feature import corner_harris, corner_peaks
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def corner_detector(img):
    cimg = corner_harris(img,method='k', k=0.2)
    # img2 = np.array(img, np.uint8)
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp, des = sift.detectAndCompute(img2, None)
    return cimg





