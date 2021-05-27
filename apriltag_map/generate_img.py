#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-03-10 18:23:19
@LastEditors  : LI Jinjie
@LastEditTime : 2020-03-10 18:23:19
@Units        : Meters
@Description  : This python script can generate an image with apriltags arranged as you wish
@Dependencies : 注意opencv，左上角是图像原点。现在是
@NOTICE       : None
'''

import os
import numpy as np
import cv2

HEIGHT = 2160  # pixels of the whole map's height
WIDTH = 1920  # pixels of the whole map's width

METER_FLAG = False  # use meter or pixels
BORDER = 0.2  # The length of one BLACK side. Unit: m 如果边长是1m，则这里就是比例。
PIXELS_P_METER = 1000  # pixels per meter. 确认BORDER×PIXELS_P_METER是PIXELS_P_SIDE的整数倍

PIXELS_P_SIDE = 10  # 原始二维码一边有几个像素

if (BORDER * PIXELS_P_METER) % PIXELS_P_SIDE != 0:
    raise ValueError('二维码没有被整数倍放大，请检查')

DISTANCE = 0.2  # The horizontal distance between 2 apriltags
# DISTANCE_H = 1.0  # The verstical distance between 2 apriltags
NUM_H = 9  # The number of tags in height side. Must be an odd number!
NUM_W = 9  # The number of tags in width side. Must be an odd number!
# NUM_H = 5  # The number of tags in height
WIDTH_OUT = 0.5  # 最外边留白的宽度, meter

dirPath = "./Pictures/"

if __name__ == "__main__":
    if NUM_H % 2 == 0 or NUM_W % 2 == 0:
        raise ValueError('一边的二维码数量NUM必须为奇数！请检查')

    if METER_FLAG is True:
        widthAllM = (DISTANCE * (NUM_W - 1) + BORDER * 1 + WIDTH_OUT * 2)  # unit: m
        widthAllP = int(widthAllM * PIXELS_P_METER)

        heightAllM = (DISTANCE * (NUM_H - 1) + BORDER * 1 + WIDTH_OUT * 2)  # unit: m
        heightAllP = int(heightAllM * PIXELS_P_METER)
    else:
        widthAllP = WIDTH
        heightAllP = HEIGHT

    borderPix = int(BORDER * PIXELS_P_METER)
    AllNum = NUM_H * NUM_W
    imgNames = os.listdir(dirPath)
    imgNames.sort()
    if imgNames[0] != 'tag36_11_00000.png':
        raise ValueError(
            'There is other files in the Picture folder. Please check!')

    # ================制作图片=============
    imgNoTag = np.ones((heightAllP, widthAllP), dtype=np.uint8)
    imgNoTag = imgNoTag * 127

    center = np.array([heightAllP / 2, widthAllP / 2], dtype=int)

    origin = np.array([center[0] + (NUM_H - 1) / 2 * DISTANCE * PIXELS_P_METER + BORDER / 2 * PIXELS_P_METER - 1,
                       center[1] - (NUM_W - 1) / 2 * DISTANCE * PIXELS_P_METER - BORDER / 2 * PIXELS_P_METER],dtype=int)
    # origin = np.array((widthAllP - 1 - WIDTH_OUT*PIXELS_P_METER,
    #                    WIDTH_OUT*PIXELS_P_METER))  # 是第一个二维码左下角的位置，左下为原点

    for i, name in enumerate(imgNames):
        if i == AllNum:
            break
        tag = cv2.imread(dirPath + name, flags=cv2.IMREAD_GRAYSCALE)
        # img = img[1:9, 1:9]  # remove the white border
        tagBig = cv2.resize(tag, (borderPix, borderPix),
                            interpolation=cv2.INTER_NEAREST)

        dh = int(i / NUM_W) * DISTANCE * PIXELS_P_METER  # 纵向：移动了NUM_W个二维码，往上走一个
        dw = int(i % NUM_W) * DISTANCE * PIXELS_P_METER  # 横向：往右最多移动NUM_W个二维码
        coor_lb = np.array((-dh, dw), dtype=int) + origin
        coor_rt = coor_lb + np.array((-borderPix + 1, borderPix - 1), dtype=int)
        coor_lt = np.array((coor_rt[0], coor_lb[1]), dtype=int)

        imgNoTag[coor_lt[0]:coor_lt[0] + borderPix,
        coor_lt[1]:coor_lt[1] + borderPix] = tagBig

        # if i >= 62 + 1:
        #     print("coor_lb", coor_lb)
        #     print("coor_rt", coor_rt)
        #     print("coor_lt", coor_lt)
        #     print(i)
        #     cv2.imshow('imgTag', imgNoTag)
        #     cv2.waitKey(0)

    # # 检查中心二维码的位置是否与图片的中心重叠
    # imgNoTag[int(heightAllP / 2 - 1):int(heightAllP / 2 + 1), :] = 0
    # imgNoTag[:, int(widthAllP / 2 - 1):int(widthAllP / 2 + 1)] = 0

    # imgNoTag[widthAllP - 10:widthAllP - 5, 5:10] = 255  # a small tag

    imgTag = imgNoTag
    cv2.imshow('imgFinal', imgTag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('apriltagMap_%dx%d_%.4fm.png' %
                (NUM_W, NUM_H, WIDTH), imgTag)

    # =================制作文本=================
    txtName = 'apriltagMapInfo_%dx%d_%.4fm.txt' % (NUM_W, NUM_H, WIDTH)
    fp = open(txtName, 'w')
    fp.write("name: 'my_bundle',\nlayout:\n  [\n")

    for i, name in enumerate(imgNames):
        originM = (-(NUM_W - 1) / 2 * DISTANCE, -(NUM_H - 1) /
                   2 * DISTANCE)  # 坐标变换，改成中间的二维码为原点
        dwM = int(i % NUM_W) * DISTANCE
        dhM = int(i / NUM_H) * DISTANCE
        if i == AllNum - 1:
            text = "    {id: %d, size: %.2f, x: %.4f, y: %.4f, z: 0.0, qw: 1.0, qx: 0.0, qy: 0.0, qz: 0.0}\n  ]" % (
                int(i), BORDER, originM[0] + dwM, originM[1] + dhM)
            fp.writelines(text)
            break
        text = "    {id: %d, size: %.2f, x: %.4f, y: %.4f, z: 0.0, qw: 1.0, qx: 0.0, qy: 0.0, qz: 0.0},\n" % (
            int(i), BORDER, originM[0] + dwM, originM[1] + dhM)
        fp.writelines(text)

    fp.close()  # 不要忘记
