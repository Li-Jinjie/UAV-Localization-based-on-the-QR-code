#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-03-07 15:13:02
@LastEditors  : LI Jinjie
@LastEditTime : 2020-03-30 17:51:18
@Units        : None
@Description  : Please read the paper: "Flexible Layouts for Fiducial Tags" by Maximilian K. .etc
@Dependencies : None
@NOTICE       : None
'''
import cv2
import numpy as np


class tags_detector:
    def __init__(self, img, small_img_shape):
        self.SMALL_IMG_SHAPE = small_img_shape
        self.


def img_preprocessing(imgOrg, SMALL_IMG_SHAPE):
    '''
    Include: resize, threshold, Morphology open
    Return: resized img, thresholded img, imgMor
    '''
    # a) Resize the image (image decimation)
    imgSmall = cv2.resize(
        imgOrg, SMALL_IMG_SHAPE[-1::-1], dst=cv2.INTER_NEAREST)

    # b) Threshold, I use Otsu's Binarization
    _, imgBlackWhite = cv2.threshold(
        imgSmall, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgBlackWhite_r = 255*np.ones(SMALL_IMG_SHAPE, dtype=np.uint8) - \
        imgBlackWhite  # reverse the color

    # c) Morphology open, to remove some noise. 好像去掉了一些细节不知道对角点精度有没有影响，再想想
    kernel = np.ones((3, 3), np.uint8)
    imgMor = cv2.morphologyEx(imgBlackWhite_r, cv2.MORPH_OPEN, kernel)
    return imgSmall, imgBlackWhite_r, imgBlackWhite, imgMor


def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if ar >= 0.8 and ar <= 1.2:
            return True, approx

    return False, 0


if __name__ == "__main__":
    # 360 Rows, 640 Columns, same with np.array.shape
    SMALL_IMG_SHAPE = (360, 640)

    # STEP 0: get image
    strFilePath = 'Raw_pictures/image2.png'   # QRcode_1.jpg  image5.png
    imgOrg = cv2.imread(strFilePath, flags=cv2.IMREAD_GRAYSCALE)

    # STEP 1: get the image coordinates of QR code
    # a) Preprocess the image
    imgSmall, imgBlackWhite_r, imgBlackWhite, imgMor = img_preprocessing(
        imgOrg, SMALL_IMG_SHAPE)

    # ==========================================
    # b) Find contours.  RETR_EXTERNAL: Only external contours.
    _, contours, _ = cv2.findContours(
        imgMor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # CHAIN_APPROX_NONE(save all points), CHAIN_APPROX_SIMPLE

    # The points in the contour is continuous, so there's no need to sort.

    # Next, we need to find the corner points. There is three ways:
    # 1. cv2.approxPolyDP() with Ramer–Douglas–Peucker algorithm 用的这个
    # 2. In cv2.findContours(), choose "CV_CHAIN_APPROX_SIMPLE"  --- low precision
    # 3. Hough transfer.
    # 4. Use Harris to calculate detect the corner points. --- similar with the paper, so I choose this way. 最好的方式是自己写Harris角点检测，只检测contour上的点.也可以SIMPLA保存角点，然后找角点邻域内最大的点保存。

    # e) Find the corner points.

    contourList = []
    cornersList = []
    for contour in contours:
        flag, corners = detect(contour)
        if flag == True:
            contourList.append(contour)
            cornersList.append(corners)
    imgFilter = np.zeros(SMALL_IMG_SHAPE, dtype=np.uint8)
    cv2.drawContours(imgFilter, contourList, -1, 255, 1)
    imgFinal = np.copy(imgSmall)
    points2 = np.array([[0, 0], [0, 80], [80, 80], [80, 0]])
    for corners in cornersList:
        h, status = cv2.findHomography(corners, points2)
        QRcode = cv2.warpPerspective(imgBlackWhite, h, (80, 80))
        _, QRcode = cv2.threshold(QRcode, 200, 255, cv2.THRESH_BINARY)
        QRcodeSmall = cv2.resize(QRcode, (8, 8), dst=cv2.INTER_NEAREST)
        cv2.imwrite('results/QRcodeSmall.jpg', QRcodeSmall)
        cv2.imshow('imgNew', QRcode,)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
