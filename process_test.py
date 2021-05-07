#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: process_test.py
Date: 2021/5/7 8:07
LastEditors: LI Jinjie
LastEditTime: 2021/5/7 8:07
Description: file content
'''

import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("result_pictures/L3_add_1.png", flags=cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("imgOrg", img)
    # cv2.waitKey(0)

    # a) Filtering?

    # b) Edge detection

    # imgEdge = cv2.Canny(img, 200, 250)
    # cv2.imshow("imgOrg", img)
    # cv2.imshow("imgEdge", imgEdge)
    # cv2.waitKey(0)
    # img[imgEdge > 100] = 127

    # cv2.imshow("imgOrg", img)
    # cv2.waitKey(0)

    # b) Threshold, I use Otsu's Binarization
    # _, imgBlackWhite = cv2.threshold(
    #     img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, imgBlackWhite = cv2.threshold(
        img, 120, 255, cv2.THRESH_BINARY)

    imgBlackWhite = 255 - imgBlackWhite
    # cv2.imshow("imgBW", imgBlackWhite)
    # cv2.waitKey(0)

    # c) Morphology open, to remove some noise. 好像去掉了一些细节不知道对角点精度有没有影响，再想想
    kernel = np.ones((3, 3), np.uint8)
    imgMor = cv2.morphologyEx(imgBlackWhite, cv2.MORPH_OPEN, kernel)

    imgMorOpen = imgMor.copy()

    kernel = np.ones((4, 4), np.uint8)  # need to adjust more carefully
    imgMor = cv2.morphologyEx(imgMor, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("imgMorOpen", imgMorOpen)
    # cv2.imshow("imgMor", imgMor)
    # cv2.waitKey(0)

    # d) find contours and contour approximation
    contours, hierarchy = cv2.findContours(imgMor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cornersList = []
    for c in contours:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) >= 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # if 0.8 <= ar <= 1.2 and (w > 20 and h > 20):
            #     cornersList.append(approx)
            if w * h > 1000:
                cornersList.append(approx)
    # black = np.zeros(img.shape, dtype=np.uint8)

    # (1) 得到每个四边形外接矩形的lt和rb坐标
    rectangleList = []
    for corners in cornersList:
        # cv2.fillConvexPoly(img, corners, 255)
        # cv2.fillPoly(black, [corners], 255)

        lt = np.array([np.min(corners[:, 0, 0]), np.min(corners[:, 0, 1])])
        rb = np.array([np.max(corners[:, 0, 0]), np.max(corners[:, 0, 1])])
        rectangleList.append(np.array([lt, rb]))

        # for i in range(corners.shape[0]):
        #     cv2.circle(img, (corners[i, 0, :].item(0), corners[i, 0, :].item(1)), 2, 255, -1)

    # cv2.imshow("imgWithPoints", img)
    # cv2.waitKey(0)

    # (2) 判断矩形的重叠
    N = len(rectangleList)
    i = 0
    while i != N:
        for j in range(i + 1, N):
            rect1 = rectangleList[i]  # 0: lt, 1: rb
            rect2 = rectangleList[j]

            # 求相交位置的坐标
            p1 = np.max([rect1[0, :], rect2[0, :]], axis=0)  # lt
            p2 = np.min([rect1[1, :], rect2[1, :]], axis=0)  # rb

            if (p2[0] > p1[0]) and (p2[1] > p1[1]):
                # add new rectangle
                lt_new = np.min([rect1[0, :], rect2[0, :]], axis=0)
                rb_new = np.max([rect1[1, :], rect2[1, :]], axis=0)

                rectangleList.append(np.array([lt_new, rb_new]))
                # delete old
                rectangleList.pop(j)  # 先删除后边的元素
                rectangleList.pop(i)
                i = 0
                N = N - 1
                break
            else:
                j = j + 1
        i = i + 1

    # # ======= to display =========
    # imgPoints = img.copy()
    # for rect in rectangleList:
    #     cv2.rectangle(imgPoints, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), 255)
    # cv2.imshow("imgWithPoints", imgPoints)
    # cv2.waitKey(0)

    # (3) 得到矩形四个点的坐标
    tagCornersList = []  # 4 * 2
    for rect in rectangleList:  # each rectangle
        tagCorners = np.zeros([4, 1, 2], dtype=np.int32)
        lt = rect[0]
        rb = rect[1]
        rt = np.array([rb.item(0), lt.item(1)])
        lb = np.array([lt.item(0), rb.item(1)])
        rect_corners = [lt, rt, rb, lb]

        for idx, pts in enumerate(rect_corners):  # each point of a rectangle
            distance_min = 9999
            for corners in cornersList:
                # if in the rectangle
                if (lt.item(0) <= corners[0, 0, 0] <= rb.item(0)) and (lt.item(1) <= corners[0, 0, 1] <= rb.item(1)):
                    for corner in corners:
                        distance = np.sum(np.abs(pts - corner))  # Manhattan Distance
                        if distance < distance_min:
                            distance_min = distance
                            tagCorners[idx, 0, :] = corner
                else:
                    continue

        tagCornersList.append(tagCorners)

    # # ======= to display =========
    # imgPolyLines = img.copy()
    # for tagCorners in tagCornersList:
    #     cv2.polylines(imgPolyLines, [tagCorners], True, 255)
    # cv2.imshow("imgWithPoints", imgPolyLines)
    # cv2.waitKey(0)
    pass

# e) Convex Hull

# d) HoughLine
# imgEdgeBW = cv2.Canny(black, 100, 200)
# # cv2.imshow("imgEdgeBW", imgEdgeBW)
# # cv2.waitKey(0)
#
# lines = cv2.HoughLines(imgEdgeBW, 1, np.pi / 180, 100)
# for i in range(lines.shape[0]):
#     rho = lines[i].item(0)
#     theta = lines[i].item(1)
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * a)
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * a)
#
#     cv2.line(img, (x1, y1), (x2, y2), 255, 2)
#
# cv2.imshow("imgWithLines", img)
# cv2.waitKey(0)
