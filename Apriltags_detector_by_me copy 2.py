#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-03-07 15:13:02
@LastEditors  : LI Jinjie
@LastEditTime : 2020-03-30 16:41:26
@Units        : None
@Description  : Please read the paper: "Flexible Layouts for Fiducial Tags" by Maximilian K. .etc
@Dependencies : None
@NOTICE       : None
'''
import cv2
import numpy as np


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
    imgBlackWhite = 255*np.ones(SMALL_IMG_SHAPE, dtype=np.uint8) - \
        imgBlackWhite  # reverse the color

    # c) Morphology open, to remove some noise. 好像去掉了一些细节不知道对角点精度有没有影响，再想想
    kernel = np.ones((3, 3), np.uint8)
    imgMor = cv2.morphologyEx(imgBlackWhite, cv2.MORPH_OPEN, kernel)
    return imgSmall, imgBlackWhite, imgMor


def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if ar >= 0.8 and ar <= 1.2:
            return True
        return False


if __name__ == "__main__":
    # 360 Rows, 640 Columns, same with np.array.shape
    SMALL_IMG_SHAPE = (360, 640)

    # STEP 0: get image
    strFilePath = 'Raw_pictures/image2.png'   # QRcode_1.jpg  image5.png
    imgOrg = cv2.imread(strFilePath, flags=cv2.IMREAD_GRAYSCALE)

    # STEP 1: get the image coordinates of QR code
    # a) Preprocess the image
    imgSmall, imgBlackWhite, imgMor = img_preprocessing(
        imgOrg, SMALL_IMG_SHAPE)

    # ==========================================
    # b) Find contours.  RETR_EXTERNAL: Only external contours.
    _, contours, _ = cv2.findContours(
        imgMor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # CHAIN_APPROX_NONE(save all points), CHAIN_APPROX_SIMPLE
    imgContours = np.copy(imgSmall)
    # cv2.drawContours(imgContours, contours, -1, 255, 1)
    # cv2.waitKey(0)
    # The points in the contour is continuous, so there's no need to sort.

    # Next, we need to find the corner points. There is three ways:
    # 1. cv2.approxPolyDP() with Ramer–Douglas–Peucker algorithm
    # 2. In cv2.findContours(), choose "CV_CHAIN_APPROX_SIMPLE"  --- low precision
    # 3. Hough transfer.
    # 4. Use Harris to calculate detect the corner points. --- similar with the paper, so I choose this way. 最好的方式是自己写Harris角点检测，只检测contour上的点.也可以SIMPLA保存角点，然后找角点邻域内最大的点保存。

    # e) Find the corner points.
    # Get solid thresholded img and calculate harris values
    imgOnlyContours = np.zeros(SMALL_IMG_SHAPE, dtype=np.uint8)
    cv2.drawContours(imgOnlyContours, contours, -1, 255, -1)
    harrisV = cv2.cornerHarris(imgOnlyContours, 2, 3, 0.04)
    # HarrisV存储的是计算出的harris值，

    # ------------------------------------
    # 这一步目前是多余的，但为了拓展自写harris计算，保留了harrisVOnContours
    # Choose harris values on the contours
    imgOnlyContours = np.zeros(SMALL_IMG_SHAPE, dtype=np.uint8)
    cv2.drawContours(imgOnlyContours, contours, -1, 1, 1)
    harrisVOnContours = harrisV * imgOnlyContours
    # ------------------------------
    contourList = []
    for contour in contours:
        if detect(contour) == True:
            contourList.append(contour)
    imgNew = np.zeros(SMALL_IMG_SHAPE, dtype=np.uint8)
    cv2.drawContours(imgNew, contourList, -1, 255, 1)
    cv2.imshow('imgNew', imgNew)
    cv2.waitKey(0)
    # --------------------------
    # find corner points in every contour
    cornersList = []
    for contour in contours:
        if detect(contour) is True:
            contourList.append(contour)
        maxHarrisV = 0
        for point in contour:
            tmp = harrisVOnContours[point[0, 1]][point[0, 0]]
            if maxHarrisV < tmp:
                maxHarrisV = tmp
        del tmp

        # Collect corner points
        pointsList = []
        for point in contour:
            row = point[0, 1]  # pay attention to the order of the coordinates
            column = point[0, 0]
            # is corner
            if harrisVOnContours[row][column] > (0.01 * maxHarrisV):
                pointsList.append(
                    (row, column, harrisVOnContours[row][column]))
            else:
                pass

        if len(pointsList) < 4:
            pass
        else:
            # Choose the point with max harrisV from several nearby points (nearby: manhattan Distance < 6)
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
            pointsAry = np.array(pointsList, dtype=int)[:, :-1]
            if len(pointsList) == 4:
                cornersList.append(pointsAry)
            else:
                # 我现在要过滤成四个点。
                cornersList.append(pointsAry)

    imgCorner = np.copy(imgSmall)
    for corners in cornersList:
        for corner in corners:
            imgCorner[corner[0]][corner[1]] = 255

    cv2.imshow('imgCorner', imgCorner)
    # cv2.imshow('imgOnlyContours', imgOnlyContours)
    # cv2.imshow('imgSmall', imgSmall)
    cv2.imwrite('results/imgCorner.jpg', imgCorner)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
