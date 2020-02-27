'''
  File name: mymosaic.py
  Author:
  Date created:
'''
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.path import Path
from scipy import ndimage
from scipy import signal
from skimage.feature import corner_peaks

import utils
from anms import anms
from corner_detector import corner_detector
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''


def warpImg(img, H):
    nr = img.shape[0]
    nc = img.shape[1]

    transVerts = np.dot(H, np.array([[0, nc - 1, nc - 1, 0],
                                     [0, 0, nr - 1, nr - 1],
                                     [1, 1, 1, 1]]))
    transVerts = transVerts / transVerts[-1, :]
    transVerts = np.around(transVerts).astype(int)
    nc_min, nc_max = np.amin(transVerts[0, :]), np.amax(transVerts[0, :])
    nr_min, nr_max = np.amin(transVerts[1, :]), np.amax(transVerts[1, :])

    xx, yy = np.meshgrid(np.arange(nc_min, nc_max), np.arange(nr_min, nr_max))
    cords = np.stack((xx.reshape(-1), yy.reshape(-1), np.ones_like(xx.reshape(-1))))  ###
    trans_cords = np.dot(np.linalg.inv(H), cords)
    trans_cords = trans_cords / trans_cords[-1, :]
    trans_cords = np.around(trans_cords).astype(int)

    queryCords = np.roll(trans_cords[0:2, :], 1, axis=0)
    I_temp = {}
    for i in range(3):  # along 3 channels
        temp = ndimage.map_coordinates(img[:, :, i], queryCords, order=1, mode='constant',
                                       cval=0.0)
        I_temp[i] = np.reshape(temp, xx.shape)
    warpedImg = np.dstack((I_temp[0], I_temp[1], I_temp[2]))

    return warpedImg, transVerts


def creatMask(Verts1, Verts2):
    nc_min, nc_max = np.amin(Verts2[0, :]), np.amax(Verts2[0, :])
    nr_min, nr_max = np.amin(Verts2[1, :]), np.amax(Verts2[1, :])
    x, y = np.meshgrid(np.arange(nc_min, nc_max + 1), np.arange(nr_min, nr_max + 1))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    tupVerts = tuple(map(tuple, Verts1[0:2, :].T))
    poly = Path(tupVerts)  # make a polygon
    grid = poly.contains_points(points)
    mask = grid.reshape(nr_max + 1 - nr_min, nc_max + 1 - nc_min)
    mask = mask.astype(int)
    mask3d = np.dstack((mask, mask, mask))

    return mask3d


def mymosaic(img_left, img_middle, img_right, H12, H32):
    nr = img_middle.shape[0]
    nc = img_middle.shape[1]

    warpedRight, transVertsRight = warpImg(img_right, H32)
    warpedLeft, transVertsLeft = warpImg(img_left, H12)
    # #

    xR_min, xR_max = np.amin(transVertsRight[0, :]), np.amax(transVertsRight[0, :])
    yR_min, yR_max = np.amin(transVertsRight[1, :]), np.amax(transVertsRight[1, :])

    xL_min, xL_max = np.amin(transVertsLeft[0, :]), np.amax(transVertsLeft[0, :])
    yL_min, yL_max = np.amin(transVertsLeft[1, :]), np.amax(transVertsLeft[1, :])

    midVerts = np.array([[0, nc - 1, nc - 1, 0],
                         [0, 0, nr - 1, nr - 1]])

    rightPachVerts = np.array([[xR_min, xR_max, xR_max, xR_min],
                               [yR_min, yR_min, yR_max, yR_max]])

    leftPachVerts = np.array([[xL_min, xL_max, xL_max, xL_min],
                              [yL_min, yL_min, yL_max, yL_max]])

    maskMiddle32 = creatMask(transVertsRight, midVerts)
    maskMiddle12 = creatMask(transVertsLeft, midVerts)
    maskRight = creatMask(midVerts, rightPachVerts)
    maskLeft = creatMask(midVerts, leftPachVerts)

    offsetX, offsetY = 4400, 2000
    npad = ((offsetY, 900), (offsetX, 2000), (0, 0))  # npad is a tuple of (n_before, n_after) for each dimension
    canvas = np.pad(np.zeros_like(img_middle), pad_width=npad, mode='constant', constant_values=0)
    canvas = np.int_(canvas)
    # canvas = np.zeros((5 * nr, 5 * nc, 3), dtype=int)  # make a canvas with coordinates

    alphaR, alphaL = 0.9, 0.1

    canvas[offsetY + yR_min:offsetY + yR_max, offsetX + xR_min:offsetX + xR_max, :] += (
            warpedRight[:, :, :] - (warpedRight[:, :, :] * (1 - alphaR) * maskRight[:-1, :-1, :])).astype(int)
    #
    canvas[offsetY + yL_min:offsetY + yL_max, offsetX + xL_min:offsetX + xL_max, :] += (
            warpedLeft[:, :, :] - (warpedLeft[:, :, :] * (1 - alphaL) * maskLeft[:-1, :-1, :])).astype(int)
    #
    canvas[offsetY:offsetY + nr, offsetX: offsetX + nc, :] += (
            img_middle[:, :, :] - img_middle[:, :, :] * alphaR * maskMiddle32 - img_middle[:, :,
                                                                                :] * alphaL * maskMiddle12).astype(int)

    ##
    plt.imshow(canvas)
    plt.show()
    img_mosaic = canvas
    return img_mosaic


if __name__ == "__main__":

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


    im3 = Image.open('right.jpg').convert('RGB')
    im2 = Image.open('middle.jpg').convert('RGB')
    im1 = Image.open('left.jpg').convert('RGB')

    img_right = np.array(im3)
    I_right = rgb2gray(img_right)
    img_middle = np.array(im2)
    I_middle = rgb2gray(img_middle)
    img_left = np.array(im1)
    I_left = rgb2gray(img_left)

    G = utils.GaussianPDF_2D(0, 1.5, 3, 3)

    I_right = signal.convolve2d(I_right, G, 'same')
    I_middle = signal.convolve2d(I_middle, G, 'same')
    I_left = signal.convolve2d(I_left, G, 'same')

    cimg_right = corner_detector(I_right)
    coords1 = corner_peaks(cimg_right, min_distance=20, threshold_rel=1e-5)
    x_right, y_right, rmax_right = anms(cimg_right, 2000)
    desc_right = feat_desc(I_right, x_right, y_right)

    cimg_middle = corner_detector(I_middle)
    coords2 = corner_peaks(cimg_middle, min_distance=20, threshold_rel=1e-5)
    x_middle, y_middle, rmax_middle = anms(cimg_middle, 2000)
    desc_middle = feat_desc(I_middle, x_middle, y_middle)

    cimg_left = corner_detector(I_left)
    coords3 = corner_peaks(cimg_left, min_distance=20, threshold_rel=1e-5)
    x_left, y_left, rmax_left = anms(cimg_left, 2000)
    desc_left = feat_desc(I_left, x_left, y_left)

    toImage = Image.new('RGB', (im1.size[0] * 2, im1.size[1]))
    toImage.paste(im3, (0, 0))
    toImage.paste(im2, (im1.size[0], 0, im1.size[0] + im2.size[0], 0 + im2.size[1]))

    plt.imshow(toImage)
    plt.plot(coords1[:, 1], coords1[:, 0], 'r.')
    plt.plot(np.array(coords2[:, 1]) + im1.size[0], np.array(coords2[:, 0]), 'r.')
    plt.show()

    plt.imshow(toImage)
    plt.plot(x_right, y_right, 'r.')
    plt.plot(np.array(x_middle) + im1.size[0], np.array(y_middle), 'r.')
    plt.show()

    match_rm1 = feat_match(desc_right, desc_middle)
    match_rm2 = feat_match(desc_middle, desc_right)
    matchIndex = np.where(match_rm1 != (-1))[0]
    x1, y1, x2, y2 = [], [], [], []
    for i in range(matchIndex.shape[0]):
        if match_rm2[match_rm1[matchIndex[i]]] == matchIndex[i]:
            x1.append(x_right[matchIndex[i]])
            y1.append(y_right[matchIndex[i]])
            x2.append(x_middle[match_rm1[matchIndex[i]]])
            y2.append(y_middle[match_rm1[matchIndex[i]]])
    x1, y1, x2, y2 = np.array(x1), np.array(y1), np.array(x2), np.array(y2)

    match_lm1 = feat_match(desc_left, desc_middle)
    match_lm2 = feat_match(desc_middle, desc_left)
    matchIndex = np.where(match_lm1 != (-1))[0]
    xl1, yl1, xl2, yl2 = [], [], [], []
    for i in range(matchIndex.shape[0]):
        if match_lm2[match_lm1[matchIndex[i]]] == matchIndex[i]:
            xl1.append(x_left[matchIndex[i]])
            yl1.append(y_left[matchIndex[i]])
            xl2.append(x_middle[match_lm1[matchIndex[i]]])
            yl2.append(y_middle[match_lm1[matchIndex[i]]])
    xl1, yl1, xl2, yl2 = np.array(xl1), np.array(yl1), np.array(xl2), np.array(yl2)

    plt.imshow(toImage)
    for i in range(x1.shape[0]):
        plt.scatter([x1[i], x2[i] + im1.size[0]], [y1[i], y2[i]], color='b')
    thresh = 1
    [H32, inlier_ind] = ransac_est_homography(x1, y1, x2, y2, thresh)
    [H12, inlier_ind2] = ransac_est_homography(xl1, yl1, xl2, yl2, thresh)
    print(inlier_ind)
    for i in inlier_ind:
        plt.plot([x1[i], x2[i] + im1.size[0]], [y1[i], y2[i]], color='r')
        plt.scatter([x1[i], x2[i] + im1.size[0]], [y1[i], y2[i]], color='r')
    plt.show()

    H12 = np.array([[-4.87657724e-04, 1.59608189e-05, 9.17082835e-01],
                    [-9.52537557e-05, - 4.19124774e-04, 3.98696132e-01],
                    [-4.97759278e-08, - 5.72458603e-09, - 2.13038506e-04]])  # 8points works

    H32 = np.array([[1.20848292e-03, 4.72858457e-05, 9.97944737e-01],
                    [-9.07753256e-05, 1.23146083e-03, 6.40410914e-02],
                    [-5.65880933e-08, -5.97729261e-08, 1.43302925e-03]])  # old

    [img_mosaic] = mymosaic(img_left, img_middle, img_right, H12, H32)
    # print(".............")
