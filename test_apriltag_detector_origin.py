#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-02-25 17:31:23
@LastEditors  : LI Jinjie
@LastEditTime : 2020-03-07 15:15:06
@Units        : None
@Description  : test solvePnP(), problems: 1. xy坐标系的符号是相反的 2. 如果一开始图像缩放了，各种参数都需要改变。
@Dependencies : None
@NOTICE       : None
'''
import math
import cv2
import numpy as np
from apriltag import apriltag
import tools as tl

H_FOV = math.pi*(65/180)  # unit: radius

imagepath = 'Raw_pictures/image5.png'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
imWidth = image.shape[1]
imHeight = image.shape[0]
cv2.imshow("1", image)
detector = apriltag("tag36h11")
detections = detector.detect(image)

# Prepare for the parameters of solvePnP, pay attention to the order of four points. In origin code, the order is 'lb-rb-rt-lt', however, in opencv, the order is 'lt-rt-rb-lb'
imagePoints = detections[1]['lb-rb-rt-lt'][::-1, :]  # transfer to lt-rt-rb-lb
worldCoor = tl.get_coordinates(detections[1]['id'])
worldPoints = worldCoor['lb-rb-rt-lt'][::-1, :]
cameraMatrix = np.zeros((3, 3), dtype=float)

fx = imWidth/(2*math.tan(H_FOV/2))
cameraMatrix[0, 0] = fx
cameraMatrix[1, 1] = fx  # in standard camera model, fy = fx
cameraMatrix[0, 2] = imWidth/2
cameraMatrix[1, 2] = imHeight/2
print('cameraMatrix = \n', cameraMatrix)

# If the vector is NULL/empty, the zero distortion coefficients are assumed.
'''
objectPoints - 世界坐标系下的控制点的坐标 3 X N
imagePoints - 在图像坐标系下对应的控制点的坐标 2 X N
cameraMatrix - 相机的内参矩阵
distCoeffs - 相机的畸变参数
flags - pnp方法
'''
retval, rvec, tvec = cv2.solvePnP(worldPoints, imagePoints, cameraMatrix, np.zeros((5)),
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)

print('retval = \n', retval)
print('rvec = \n', rvec)
print('tvec = \n', tvec)

cv2.waitKey(0)
cv2.destroyAllWindows()
