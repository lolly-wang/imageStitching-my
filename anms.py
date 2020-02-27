'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

import numpy as np
from corner_detector import corner_detector
from skimage.feature import corner_harris, corner_peaks


def anms(cimg, max_pts):
    coords = corner_peaks(cimg, min_distance=20, threshold_rel=1e-5)
    # coords = corner_peaks(cimg, min_distance=15, threshold_rel=1e-5) #cloud e-8
    # peaks = np.zeros_like(cimg)
    # peaks[coords[:, 0], coords[:, 1]] = 1
    print('corner_peaks detect: ' + str(coords.shape[0]) + ' corners')

    minR = np.zeros(coords.shape[0])

    # Create matrix of sorted values and their coordinates
    val = cimg[coords[:, 0], coords[:, 1]]
    x = coords[:, 0]
    mat = np.stack((cimg[coords[:, 0], coords[:, 1]], coords[:, 0], coords[:, 1])).T
    mat = mat[mat[:, 0].argsort()]

    for i in range(mat.shape[0]):
        j = i + 1
        while (j < coords.shape[0]) and (mat[j, 0] < 1.5 * mat[i, 0]):
            j += 1
        else:
            cnt = j - i - 1
            if cnt == 0:
                minR[i] = np.inf
            else:
                neighbors = mat[i + 1:i + 1 + cnt, 1:-1]
                ssd = np.sum((np.tile(mat[i, 1:-1], (cnt, 1)) - neighbors) ** 2, axis=1)
                minR[i] = np.amin(ssd)
    descR = -np.sort(-minR)
    rmax = descR[max_pts]
    indx = np.where(minR > rmax)[0]
    x, y = mat[indx, 2], mat[indx, 1]
    x = np.int_(x)
    y = np.int_(y)
    # print('anms detect: ' + str(x.shape[0]) + ' corners')

    return x, y, rmax ** 0.5
