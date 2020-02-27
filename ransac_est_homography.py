'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''
import numpy as np
import random
from est_homography import est_homography

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''


def ransac_est_homography(x1, y1, x2, y2, thresh):
    # Your Code Here
    inliers = [list() for i in range(100000)]
    count = []
    for i in range(100000):#1000000
        randomIndices = np.array(random.sample(range(x1.shape[0]), 4))
        H = est_homography(x1[randomIndices], y1[randomIndices], x2[randomIndices], y2[randomIndices])
        # index = list(range(x1.shape[0]))
        # index = list(filter(lambda x: x not in list(randomIndices), index))
        coords = np.vstack((x1, y1, np.ones(x1.shape[0])))
        [x2_est, y2_est, z2_est] = np.dot(H, coords)
        x2_est, y2_est = x2_est / z2_est, y2_est / z2_est
        ssd = np.square(x2 - x2_est) + np.square(y2 - y2_est)
        inliers[i].extend(list(np.where(ssd < thresh)[0]))
        count.append(len(inliers[i]))
    count = np.array(count)
    ind = int(np.argmax(count))
    inlier_ind = np.array(inliers[ind])
    H = est_homography(x1[inlier_ind], y1[inlier_ind], x2[inlier_ind], y2[inlier_ind])

    # return H#, inliers
    print(H)
    return H, inlier_ind
