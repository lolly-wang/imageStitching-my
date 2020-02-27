'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

import numpy as np
from findDerivatives import findDerivatives
import cv2

def feat_desc(img, x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    # points = list(zip(x,y))
    # keypoints = [cv2.KeyPoint(m[1], m[0], 1) for m in points]
    #
    # img2 = np.array(img, np.uint8)
    # sift = cv2.xfeatures2d.SIFT_create()
    # keypoints, des = sift.compute(img2, keypoints)
    # descs = np.transpose(des)
    H, W = img.shape
    N = x.shape[0]
    descs = np.zeros((64, N))
    mag, _, _, ori = findDerivatives(img)
    temp = np.zeros((64, 25))
    for n in range(N):
        i, j = y[n], x[n]
        ii, jj = np.meshgrid(np.arange((i - 20), (i + 20)), np.arange((j - 20), (j + 20)))
        ii = np.clip(ii, 0, H - 1)
        jj = np.clip(jj, 0, W - 1)
        bigPatch = mag[ii, jj]

        uu, vv = np.meshgrid(np.arange(40), np.arange(40))
        uu = uu[::5, ::5]
        vv = vv[::5, ::5]
        cnt = 0
        for p in range(5):
            for q in range(5):
                temp[:, cnt] = bigPatch[uu + p, vv + q].reshape(-1)
                cnt += 1
        descs[:, n] = np.amax(temp, axis=1)
        # descs[:, n] -= np.mean(descs[:, n])
        # descs[:, n]/= np.var(descs[:, n])**0.5
    return descs
