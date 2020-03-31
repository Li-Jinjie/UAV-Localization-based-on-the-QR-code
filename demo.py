#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-03-31 10:18:58
@LastEditors  : LI Jinjie
@LastEditTime : 2020-03-31 10:21:16
@Units        : None
@Description  : file content
@Dependencies : None
@NOTICE       : None
'''

from Apriltags_detector_by_me import tags_detector
import cv2

# 360 Rows, 640 Columns, same with np.array.shape
SMALL_IMG_SHAPE = (360, 640)

# STEP 0: get image
strFilePath = 'Raw_pictures/image5.png'   # QRcode_1.jpg  image5.png
imgOrg = cv2.imread(strFilePath, flags=cv2.IMREAD_GRAYSCALE)
detector = tags_detector()
flag, results = detector.detect(imgOrg, SMALL_IMG_SHAPE)

if flag == True:
    for i, result in enumerate(results):
        print(result)
    print(str(len(results)) + ' apriltags are detected in total!')
else:
    print('No apriltag is detected!')
