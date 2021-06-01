#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: apriltags_detector_findContour.py
Date: 2021/5/7 8:07
LastEditors: LI Jinjie
LastEditTime: 2021/5/7 8:07
Description: a new AprilTag detector class
'''

import cv2
import numpy as np
import math
from tag36h11 import tag36h11_create


class TagsDetector:
    cornersList = []
    tagsList = []
    resultList = []  # id, hamming_distance, rotate_degree
    tag_flag = True

    # used to decode
    tag36h11List = []
    bit_x = []
    bit_y = []

    def __init__(self):
        self.tag36h11List, self.bit_x, self.bit_y = tag36h11_create()
        self.reverse_flag = -1
        self.pass_flag = 1

    def detect(self, img):
        resultList = []  # id, hamming_distance, rotate_degree  should be refresh every time
        # imgBW_adap = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
        # imgBW_adap = 255 - imgBW_adap    # 自适应阈值效果很差。
        # cv2.imshow("imgBW_adap", imgBW_adap)

        # 求中位数
        median_value = np.median(img)
        mean_value = np.mean(img)
        _, imgBlackWhite = cv2.threshold(img, median_value, 255, cv2.THRESH_BINARY)  # extremely critical

        # _, imgBlackWhite = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)  # extremely critical
        if self.reverse_flag == 1:
            imgBlackWhite = 255 - imgBlackWhite
        # cv2.imshow("imgBW", imgBlackWhite)
        # cv2.waitKey(0)

        imgBlackWhite = cv2.medianBlur(imgBlackWhite, 5)

        # c) Morphology open, to remove some noise. 好像去掉了一些细节不知道对角点精度有没有影响，再想想
        kernel = np.ones((4, 4), np.uint8)
        imgMor = cv2.morphologyEx(imgBlackWhite, cv2.MORPH_OPEN, kernel)

        imgMorOpen = imgMor.copy()

        kernel = np.ones((10, 10), np.uint8)  # need to adjust more carefully
        imgMor = cv2.morphologyEx(imgMor, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("imgBW", imgBlackWhite)
        # cv2.imshow("imgMorOpen", imgMorOpen)
        cv2.imshow("imgMorClose", imgMor)
        cv2.waitKey(0)

        # # HoughLines test
        # imgEdge = cv2.Canny(imgMor, 50, 150, apertureSize=3)
        # lines = cv2.HoughLines(imgEdge, 1, np.pi / 180, 120, None, 0, 0)
        #
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        #         cv2.line(imgBlackWhite, pt1, pt2, 127, 3, cv2.LINE_AA)
        # cv2.imshow("Canny", imgEdge)
        # cv2.imshow("HoughLines", imgBlackWhite)
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

            # ====== to display ========
            for i in range(corners.shape[0]):
                cv2.circle(img, (corners[i, 0, :].item(0), corners[i, 0, :].item(1)), 2, 255, -1)
        cv2.imshow("imgWithCorners", img)
        cv2.waitKey(0)

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
        # cv2.imshow("imgWithRectangles", imgPoints)
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
                    if (lt.item(0) <= corners[0, 0, 0] <= rb.item(0)) and (
                            lt.item(1) <= corners[0, 0, 1] <= rb.item(1)):
                        for corner in corners:
                            distance = np.sum(np.abs(pts - corner))  # Manhattan Distance
                            if distance < distance_min:
                                distance_min = distance
                                tagCorners[idx, 0, :] = corner
                    else:
                        continue

            tagCornersList.append(tagCorners)
        # ======= to display =========
        # imgPolyLines = img.copy()
        # for tagCorners in tagCornersList:
        #     cv2.polylines(imgPolyLines, [tagCorners], True, 255)
        # cv2.imshow("imgWithPoints", imgPolyLines)
        # cv2.waitKey(0)

        # Perspective transform
        self.tagsList = self._perspective_transform(imgBlackWhite, tagCornersList)

        # STEP 2 : apriltag decoding
        # a) save data of tag36h11 in __init__()
        for i, tag in enumerate(self.tagsList):
            # b) rotate the img and get four code
            intCodeList = self._rotate_get_int(tag, self.bit_x, self.bit_y)
            # c) calculate the hamming distance and find the minimal one
            hamming, id, rotate_dgree = self._find_min_hamming(intCodeList, self.tag36h11List)
            # d) filter invalid code and append result
            if hamming < 4:
                lt_rt_rd_ld = np.rot90(tagCornersList[i], rotate_dgree / 90)
                resultList.append(
                    {'id': id, 'hamming': hamming, 'lt_rt_rd_ld': lt_rt_rd_ld})

        # ======= to display =========
        imgFinal = img.copy()
        for tagCorners in tagCornersList:
            cv2.polylines(imgFinal, [tagCorners], True, 255)
        for result in resultList:
            text = "id:" + str(result["id"]) + " hamming:" + str(result["hamming"])
            org = (result["lt_rt_rd_ld"][0, :].item(0), result["lt_rt_rd_ld"][0, :].item(1))
            cv2.putText(imgFinal, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)

        cv2.imshow("imgWithResults", imgFinal)
        cv2.waitKey(0)

        # STEP 3 : update the flag
        if len(resultList) == 0:
            self.tag_flag = False
        else:
            self.tag_flag = True
            self.reverse_flag = -1 * self.reverse_flag  # 下一张是反转的检测的
            self.pass_flag = - self.pass_flag  # 去除下一张

        return self.tag_flag, resultList

    def _perspective_transform(self, imgMor, cornersList):

        tagsList = []
        sizePixel = 80
        points2 = np.array([[0, 0], [sizePixel, 0], [sizePixel, sizePixel], [0, sizePixel]])  # lt_rt_rd_ld
        for corners in cornersList:
            # cv2.polylines(imgMor, [corners], True, 127)
            # cv2.imshow("imgWithPoints", imgMor)
            # cv2.waitKey(0)

            corners = corners.squeeze()
            h, status = cv2.findHomography(corners, points2)
            QRcode = cv2.warpPerspective(imgMor, h, (sizePixel, sizePixel))

            # cv2.imshow("QRcode", QRcode)
            # cv2.waitKey(0)

            # kernel = np.ones((2, 2), np.uint8)  # need to adjust more carefully
            # QRMor = cv2.morphologyEx(QRcode, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow("QRMor", QRMor)
            # cv2.waitKey(0)

            QRBlur = cv2.medianBlur(QRcode, 3)
            # cv2.imshow("QRBlur", QRBlur)
            # cv2.waitKey(0)

            # QRBlur = cv2.medianBlur(QRBlur, 3)
            # cv2.imshow("QRBlur", QRBlur)
            # cv2.waitKey(0)

            # _, QRcode = cv2.threshold(
            #     QRcode, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            QRcodeSmall = cv2.resize(QRBlur, (8, 8), dst=cv2.INTER_NEAREST)
            _, QRcodeSmall = cv2.threshold(QRcodeSmall, 0, 255, cv2.THRESH_OTSU)
            if (np.sum(QRcodeSmall[0, :-1]) + np.sum(QRcodeSmall[:-1, -1]) + np.sum(QRcodeSmall[1:, 0]) + np.sum(
                    QRcodeSmall[-1, 1:])) / 28 > 127:
                QRcodeSmall = 255 - QRcodeSmall

            # cv2.imshow("QRcodeSmall", QRcodeSmall)
            # cv2.waitKey(0)

            tagsList.append(QRcodeSmall)
        return tagsList

    def _rotate_get_int(self, tag, bit_x, bit_y):
        intCodeList = []
        for i in range(4):
            tag = np.rot90(tag, i)
            code = '0b'
            for j in range(36):
                if tag[bit_y[j], bit_x[j]] < 100:
                    code = code + '0'
                else:
                    code = code + '1'
            intCode = int(code, 2)
            intCodeList.append(intCode)
        return intCodeList

    def _find_min_hamming(self, intCodeList, tag36h11List):
        hammingMin = 36
        idMin = 0
        rotate_dgree = 0
        for intCode in intCodeList:
            hammingMinLocal = 36
            idMinLocal = 0
            for iD, tagCode in enumerate(tag36h11List):
                s = str(bin(intCode ^ tagCode))
                hamming = 0
                for i in range(2, len(s)):
                    if int(s[i]) == 1:
                        hamming += 1
                if hammingMinLocal > hamming:
                    hammingMinLocal = hamming
                    idMinLocal = iD
                if hammingMinLocal == 0:
                    break
            if hammingMin > hammingMinLocal:
                hammingMin = hammingMinLocal
                idMin = idMinLocal
            if hammingMin == 0:
                break
            rotate_dgree += 90

        return hammingMin, idMin, rotate_dgree


if __name__ == '__main__':
    # img = cv2.imread("receiver_pictures/L3_add_1.png", flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("receiver_pictures/0517_bw_120fps_90degree_720p.avi_lab_2.png", flags=cv2.IMREAD_GRAYSCALE)
    detector = TagsDetector()
    flag, results = detector.detect(img)

    if flag == True:
        for i, result in enumerate(results):
            print(result)
        print(str(len(results)) + ' apriltags are detected in total!')
    else:
        print('No apriltag is detected!')
