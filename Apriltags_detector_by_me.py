#!/usr/bin/env python
# coding=utf-8
'''
@Author       : LI Jinjie
@Date         : 2020-03-07 15:13:02
@LastEditors  : LI Jinjie
@LastEditTime : 2020-03-31 10:09:49
@Units        : None
@Description  : Please read the paper: "Flexible Layouts for Fiducial Tags" by Maximilian K. .etc
@Dependencies : None
@NOTICE       : None
'''
import cv2
import numpy as np
from tag36h11 import tag36h11_create


class tags_detector:
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

    def detect(self, img, small_img_shape):
        # STEP 0: get img
        self.SMALL_IMG_SHAPE = small_img_shape
        self.imgOrg = img

        # STEP 1: apriltag detecting
        # a) Preprocess the image
        self.imgSmall, self.imgBlackWhite, self.imgMor = self.img_preprocessing(
            self.imgOrg, self.SMALL_IMG_SHAPE)

        # b) Find contours.
        _, self.contours, _ = cv2.findContours(
            self.imgMor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # CHAIN_APPROX_SIMPLE: save necessary points; RETR_EXTERNAL: Only external contours.
        # The points in the contour is continuous, so there's no need to sort.

        # c) Find corners
        # Next, we need to find the corner points. There are three ways:
        # 1. cv2.approxPolyDP() with Ramer–Douglas–Peucker algorithm. Simplest, so I choose this method.
        # 2. In cv2.findContours(), choose "CV_CHAIN_APPROX_SIMPLE"  --- low precision
        # 3. Hough transfer.
        # 4. Use Harris to calculate detect the corner points.
        self.cornersList = self.find_corners(self.contours)

        # d) Perspective transform
        self.tagsList = self.perspective_transform(
            self.imgBlackWhite, self.cornersList)

        # STEP 2 : apriltag decoding
        # a) save data of tag36h11 in __init__()
        for i, tag in enumerate(self.tagsList):
            # b) rotate the img and get four code
            intCodeList = self.rotate_get_int(tag, self.bit_x, self.bit_y)
            # c) calculate the hamming distance and find the minimal one
            hamming, id, rotate_dgree = self.find_min_hamming(
                intCodeList, self.tag36h11List)
            # d) filter invalid code and append result
            if hamming < 8:
                lt_rt_rd_ld = np.rot90(self.cornersList[i], rotate_dgree/90)
                self.resultList.append(
                    {'id': id, 'hamming': hamming, 'lt_rt_rd_ld': lt_rt_rd_ld})

        # STEP 3 : update the flag
        if len(self.resultList) == 0:
            self.tag_flag = False
        else:
            self.tag_flag = True

        return self.tag_flag, self.resultList

    def img_preprocessing(self, imgOrg, SMALL_IMG_SHAPE):
        '''
        Include: resize, threshold, Morphology open
        Return: resized img, thresholded img, imgMor
        '''
        # a) Resize the image (image decimation)
        imgSmall = cv2.resize(
            imgOrg, self.SMALL_IMG_SHAPE[-1::-1], dst=cv2.INTER_NEAREST)

        # b) Threshold, I use Otsu's Binarization
        _, imgBlackWhite = cv2.threshold(
            imgSmall, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        imgBlackWhite_r = 255*np.ones(SMALL_IMG_SHAPE, dtype=np.uint8) - \
            imgBlackWhite  # reverse the color

        # c) Morphology open, to remove some noise. 好像去掉了一些细节不知道对角点精度有没有影响，再想想
        kernel = np.ones((3, 3), np.uint8)
        imgMor = cv2.morphologyEx(imgBlackWhite_r, cv2.MORPH_OPEN, kernel)
        return imgSmall, imgBlackWhite, imgMor

    def find_corners(self, contours):
        cornersList = []
        for contour in contours:
            flag, corners = self.detect_shape(contour)
            if flag == True:
                cornersList.append(corners)
        return cornersList

    def detect_shape(self, c):
        # initialize the shape name and approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if ar >= 0.8 and ar <= 1.2 and w > 20 and h > 20:
                return True, approx

        return False, 0

    def perspective_transform(self, imgBlackWhite, cornersList):
        tagsList = []
        points2 = np.array([[0, 0], [0, 80], [80, 80], [80, 0]])
        for corners in cornersList:
            h, status = cv2.findHomography(corners, points2)
            QRcode = cv2.warpPerspective(imgBlackWhite, h, (80, 80))
            _, QRcode = cv2.threshold(QRcode, 200, 255, cv2.THRESH_BINARY)
            QRcodeSmall = cv2.resize(QRcode, (8, 8), dst=cv2.INTER_NEAREST)
            tagsList.append(QRcodeSmall)
        return tagsList

    def rotate_get_int(self, tag, bit_x, bit_y):
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

    def find_min_hamming(self, intCodeList, tag36h11List):
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
                    if int(s[i]) is 1:
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

    # def bin_to_int()


if __name__ == "__main__":
    # 360 Rows, 640 Columns, same with np.array.shape
    SMALL_IMG_SHAPE = (360, 640)

    # STEP 0: get image
    strFilePath = 'Raw_pictures/QRcode_1.jpg'   # QRcode_1.jpg  image5.png
    imgOrg = cv2.imread(strFilePath, flags=cv2.IMREAD_GRAYSCALE)
    detector = tags_detector()
    flag, results = detector.detect(imgOrg, SMALL_IMG_SHAPE)

    if flag == True:
        for i, result in enumerate(results):
            print(result)
        print(str(len(results)) + ' apriltags are detected in total!')
    else:
        print('No apriltag is detected!')
