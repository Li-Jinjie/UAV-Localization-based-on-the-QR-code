#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 1970-01-01 08:00:00
@LastEditors  : LI Jinjie
@LastEditTime : 2020-03-10 18:23:19
@Units        : Meters
@Description  : This python script can generate an image with apriltags arranged as you wish
@Dependencies : None
@NOTICE       : None
'''

import os
import numpy as np
import cv2


BORDER = 0.3  # The length of one BLACK side. Unit: m
PIXELS_P_METER = 240  # pixels per meter. 确认BORDER×PIXELS_P_METER是PIXELS_P_SIDE的整数倍
PIXELS_P_SIDE = 10  # 原始二维码一边有几个像素

DISTANCE = 0.9  # The horizontal distance between 2 apriltags
# DISTANCE_H = 1.0  # The verstical distance between 2 apriltags
NUM = 9  # The number of tags in one side. 必须是奇数！
# NUM_H = 5  # The number of tags in height
WIDTH_OUT = 0.5  # 最外边留白的宽度


dirPath = "Pictures/"

if __name__ == "__main__":
    if NUM % 2 == 0:
        raise ValueError('一边的二维码数量NUM必须为奇数！请检查')

    borderPix = int(BORDER * PIXELS_P_METER)
    widthAllM = (DISTANCE * (NUM-1) + BORDER * 1 + WIDTH_OUT * 2)  # unit: m
    AllNum = NUM * NUM
    widthAllP = int(widthAllM * PIXELS_P_METER)

    imgNames = os.listdir(dirPath)
    imgNames.sort()
    if imgNames[0] != 'tag36_11_00000.png':
        raise ValueError(
            'There is other files in the Picture folder. Please check!')

    # ================制作图片=============
    imgWhite = np.ones((widthAllP, widthAllP), dtype=np.uint8)
    imgWhite = imgWhite * 255

    origin = np.array((widthAllP - 1 - WIDTH_OUT*PIXELS_P_METER,
                       WIDTH_OUT*PIXELS_P_METER))  # 是第一个二维码左下角的位置，左下为原点

    for i, name in enumerate(imgNames):
        if i == AllNum:
            break
        img = cv2.imread(dirPath + name, flags=cv2.IMREAD_GRAYSCALE)
        img = img[1:9, 1:9]   # remove the white border
        imgBig = cv2.resize(img, (borderPix, borderPix),
                            interpolation=cv2.INTER_NEAREST)

        dw = int(i % NUM) * DISTANCE * PIXELS_P_METER
        dh = int(i / NUM) * DISTANCE * PIXELS_P_METER
        coor_lb = np.array((-dh, dw)) + origin
        coor_rt = coor_lb + np.array((-borderPix+1, borderPix-1))
        coor_lt = np.array((coor_rt[0], coor_lb[1]), dtype=int)

        imgWhite[coor_lt[0]:coor_lt[0]+borderPix,
                 coor_lt[1]:coor_lt[1]+borderPix] = imgBig

    # # 检查中心二维码的位置是否与图片的中心重叠
    # imgWhite[int(widthAllP/2-1):int(widthAllP/2+1), :] = 0
    # imgWhite[:, int(widthAllP/2-1):int(widthAllP/2+1)] = 0

    imgWhite[widthAllP-10:widthAllP-5, 5:10] = 0
    cv2.imshow('imgFinal', imgWhite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('apriltagMap_%dx%d_%.4fm.png' %
                (NUM, NUM, widthAllM), imgWhite)

    # =================制作文本=================
    txtName = 'apriltagMapInfo_%dx%d_%.4fm.txt' % (NUM, NUM, widthAllM)
    fp = open(txtName, 'w')
    fp.write("name: 'my_bundle',\nlayout:\n  [\n")

    for i, name in enumerate(imgNames):
        originM = (-(NUM-1)/2*DISTANCE, -(NUM-1) /
                   2*DISTANCE)  # 坐标变换，改成中间的二维码为原点
        dwM = int(i % NUM) * DISTANCE
        dhM = int(i / NUM) * DISTANCE
        if i == AllNum-1:
            text = "    {id: %d, size: %.2f, x: %.4f, y: %.4f, z: 0.0, qw: 1.0, qx: 0.0, qy: 0.0, qz: 0.0}\n  ]" % (
                int(i), BORDER, originM[0]+dwM, originM[1]+dhM)
            fp.writelines(text)
            break
        text = "    {id: %d, size: %.2f, x: %.4f, y: %.4f, z: 0.0, qw: 1.0, qx: 0.0, qy: 0.0, qz: 0.0},\n" % (
            int(i), BORDER, originM[0]+dwM, originM[1]+dhM)
        fp.writelines(text)

    fp.close()  # 不要忘记
