#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-03-07 15:13:02
@LastEditors  : LI Jinjie
@LastEditTime : 2020-03-08 10:45:49
@Units        : None
@Description  : Please read the paper: "Flexible Layouts for Fiducial Tags" by Maximilian K. .etc
@Dependencies : None
@NOTICE       : None
'''
import cv2
import numpy as np
from Apriltags_detector_by_me import img_preprocessing


# 360 Rows, 640 Columns, same with np.array.shape
SMALL_IMG_SHAPE = (360, 640)

# STEP 0: get image
strFilePath = 'Raw_pictures/QRcode_1.jpg'   # QRcode_1.jpg  image5.png
imgOrg = cv2.imread(strFilePath, flags=cv2.IMREAD_GRAYSCALE)

# STEP 1: get the image coordinates of QR code
# a) Preprocess the image
imgSmall, imgBlackWhite, imgMor = img_preprocessing(
    imgOrg, SMALL_IMG_SHAPE)


# ==========================================
# b) Find contours.  RETR_EXTERNAL: Only external contours.
_, contours, _ = cv2.findContours(
    imgMor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE(save all points)
imgContours = np.copy(imgSmall)
cv2.drawContours(imgContours, contours, -1, 255, 1)

imgOnlyContours = np.zeros(SMALL_IMG_SHAPE, dtype=np.uint8)
cv2.drawContours(imgOnlyContours, contours, -1, 255, -1)

# The points in the contour is continuous, so there's no need to sort.

# Next, we need to find the corner points. There is three ways:
# 1. cv2.approxPolyDP() with Ramer–Douglas–Peucker algorithm
# 2. In cv2.findContours(), choose "CV_CHAIN_APPROX_SIMPLE"  --- low precision
# 3. Hough transfer.
# 4. Use Harris to calculate detect the corner points. --- similar with the paper, so I choose this way. 最好的方式是自己写Harris角点检测，只检测contour上的点.也可以SIMPLA保存角点，然后找角点邻域内最大的点保存。

# e) Find the corner points.
# Get harris values
harrisV = cv2.cornerHarris(imgOnlyContours, 2, 3, 0.04)

# Choose big enough(1) harris values on the contours(2)
imgOnlyContours = np.zeros(SMALL_IMG_SHAPE, dtype=np.uint8)
cv2.drawContours(imgOnlyContours, contours, -1, 1, 1)
harrisVOnContours = harrisV * imgOnlyContours  # (2)

harrisVMax = harrisVOnContours.max()
harrisVOnContours[harrisVOnContours < 0.05*harrisVMax] = 0  # (1)

# imgSmall[harrisVOnContours != 0] = 255
# cv2.imshow('imgSmall', imgSmall)
# cv2.waitKey(0)

# find corner points in every contour
cornersList = []
for contour in contours:

    # Collect corner points
    pointsList = []
    for point in contour:
        row = point[0, 1]  # 这里的返回值好像得反一下才对
        column = point[0, 0]
        if harrisVOnContours[row][column] != 0:  # is corner
            # save the coordinate and value
            pointsList.append((row, column, harrisVOnContours[row][column]))
        else:
            pass

    # Choose the point with max harrisV from several nearby points
    i = 0
    while(i != len(pointsList)-1):
        mhtDist = abs(pointsList[i][0]-pointsList[i+1][0]) + \
            abs(pointsList[i][1]-pointsList[i+1][1])
        if mhtDist < 6:
            if pointsList[i][2] < pointsList[i+1][2]:
                del pointsList[i]
            else:
                del pointsList[i]
            i = i-1
        else:
            pass

        i = i+1

    if len(pointsList) < 4:
        pass
    else:
        pointsAry = np.array(pointsList, dtype=np.uint8)[:, :-1]
        if len(pointsList) == 4:
            cornersList.append(pointsAry)
        else:
            # 我现在要过滤成四个点。
            cornersList.append(pointsAry)


cv2.imshow('imgOnlyContours', imgOnlyContours)
cv2.imshow('imgContours', imgContours)
cv2.imshow('imgSmall', imgSmall)
cv2.waitKey(0)
cv2.destroyAllWindows()
