#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-02-25 17:31:23
@LastEditors  : LI Jinjie
@LastEditTime : 2020-02-29 18:17:36
@Units        : None
@Description  : file content
@Dependencies : None
@NOTICE       : None
'''

import cv2
import numpy as np
from apriltag import apriltag
import tools as tl

imagepath = 'image5.png'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
cv2.imshow("1", image)
detector = apriltag("tag36h11")
detections = detector.detect(image)
image_points = detections[0]['lb-rb-rt-lt']
world_coor = tl.get_coordinates(detections[0]['id'])
world_points = world_coor['lb-rb-rt-lt']
# cv2.solvePnP(world_points, image_points, flags='SOLVEPNP_IPPE_SQUARE')
'''
objectPoints - 世界坐标系下的控制点的坐标
imagePoints - 在图像坐标系下对应的控制点的坐标
cameraMatrix - 相机的内参矩阵
'''

cv2.waitKey(0)
cv2.destroyAllWindows()
