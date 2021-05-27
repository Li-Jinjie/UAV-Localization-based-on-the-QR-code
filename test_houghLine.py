#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: test_houghLine.py
Date: 2021/5/27 17:28
LastEditors: LI Jinjie
LastEditTime: 2021/5/27 17:28
Description: file content
'''
import cv2
import numpy as np
import math

imgMor = cv2.imread("receiver_pictures/sub_close.png", cv2.IMREAD_GRAYSCALE)
imgHoughLines = imgMor.copy()
imgHoughLinesP = imgMor.copy()

# 边缘检测
imgEdge = cv2.Canny(imgMor, 50, 150, apertureSize=3)
# imgEdge = cv2.Laplacian(imgMor, cv2.CV_64F).astype(np.uint8)
# imgEdge = cv2.Sobel(imgMor, cv2.CV_64F, 1, 1, ksize=5).astype(np.uint8)

# HoughLines
lines = cv2.HoughLines(imgEdge, 1, np.pi / 180, 120, None, 0, 0)
if lines is not None:
    line_length = 1500
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + line_length * (-b)), int(y0 + line_length * (a)))
        pt2 = (int(x0 - line_length * (-b)), int(y0 - line_length * (a)))
        cv2.line(imgHoughLines, pt1, pt2, 127, 2, cv2.LINE_AA)

linesP = cv2.HoughLinesP(imgEdge, 1, np.pi / 180, 100, None, 200, 60)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(imgHoughLinesP, (l[0], l[1]), (l[2], l[3]), 127, 2, cv2.LINE_AA)

cv2.imshow("Canny", imgEdge)
# cv2.imshow("Laplacian", imgLaplacian)
cv2.imshow("HoughLines", imgHoughLines)
cv2.imshow("HoughLinesP", imgHoughLinesP)
cv2.waitKey(0)
