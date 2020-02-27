'''
  File name: findDerivatives.py
  Author: Luoli Wang
  Date created: Sep 20 2019
'''

'''
  File clarification:
    Compute gradient information of the input grayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the magnitude of derivatives along x-axis
    - Output Magy: H x W matrix represents the magnitude of derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''

import numpy as np
from scipy import signal
import utils


def findDerivatives(I_gray):
    G = utils.GaussianPDF_2D(0, 1.3, 5, 5)

    Gx, Gy = np.gradient(G, axis=(1, 0))

    magx = signal.convolve2d(I_gray, Gx, 'same')
    magy = signal.convolve2d(I_gray, Gy, 'same')

    mag = np.sqrt(magx * magx + magy * magy)
    ori = np.arctan2(magy, magx, out=None)

    return mag, magx, magy, ori
