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

    # b) Threshold, I use Otsu's Binarization
    _, imgBlackWhite = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # c) Morphology open, to remove some noise. 好像去掉了一些细节不知道对角点精度有没有影响，再想想
    kernel = np.ones((3, 3), np.uint8)
    imgMor = cv2.morphologyEx(imgBlackWhite, cv2.MORPH_OPEN, kernel)

    cv2.imshow("imgMor", imgMor)
    cv2.waitKey(0)
