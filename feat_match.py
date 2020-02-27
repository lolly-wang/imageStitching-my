'''
  File name: feat_match.py
  Author:
  Date created:
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''


def feat_match(descs1, descs2):
    # Your Code Here

    # descs2 = np.transpose(np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]))
    # descs1 = np.transpose(np.array([[-3.2, -2.1], [-2.6, -1.3], [1.4, 1.0], [3.1, 2.6], [2.5, 1.0]]))
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(np.transpose(descs2))
    distances, indices = neighbors.kneighbors(np.transpose(descs1))
    # print(indices)
    # print(distances)
    unique = (distances[:, 0] / distances[:, 1]) < 0.9
    match = indices[:, 0] * unique - (1 - unique)

    return match



