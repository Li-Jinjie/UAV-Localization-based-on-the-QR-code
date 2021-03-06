#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-03-31 10:18:58
@LastEditors  : LI Jinjie
@LastEditTime : 2020-04-01 11:35:25
@Units        : None
@Description  : a demo to use apriltags_detector_by_me.py.
@Dependencies : None
@NOTICE       : None
'''

from apriltags_detector_by_me import TagsDetector
import cv2

# 360 Rows, 640 Columns, same with np.array.shape
SMALL_IMG_SHAPE = (240, 320)

# STEP 0: get image
strFilePath = 'Raw_pictures/QRcode_1.jpg'   # QRcode_1.jpg  image5.png
imgOrg = cv2.imread(strFilePath, flags=cv2.IMREAD_GRAYSCALE)
detector = TagsDetector()
flag, results = detector.detect(imgOrg, SMALL_IMG_SHAPE)

if flag == True:
    for i, result in enumerate(results):
        print(result)
    print(str(len(results)) + ' apriltags are detected in total!')
else:
    print('No apriltag is detected!')
