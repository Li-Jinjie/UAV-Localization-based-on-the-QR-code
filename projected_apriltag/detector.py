#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: detector.py
Date: 2021/6/1 19:52
LastEditors: LI Jinjie
LastEditTime: 2021/6/1 19:52
Description: a new AprilTag detector class
'''

import cv2
import numpy as np
from .tag36h11 import Tag36H11
from .preprocessing import preprocess
from .finding_corners import find_corner_using_contours
from .decoding import decode


class TagsDetector:

    def __init__(self):
        # to decode
        self.tag36h11_info = Tag36H11()

        # last frame and frame information
        self.last_frame_lightness = None
        self.height = None
        self.width = None

        # to save the results
        self.cornersList = []
        self.tagsList = []
        self.resultList = []  # id, hamming_distance, rotate_degree

        # flags
        self.tag_flag = True
        self.reverse_flag = -1
        self.pass_flag = 1

    def detect(self, img):
        """
        Detect the apriltags in the image.
        :param img:  BGR
        :return self.tag_flag, result_list:
        """
        if self.pass_flag > 0:
            self.pass_flag = - self.pass_flag
            img_lab = cv2.cvtColor(img, code=cv2.COLOR_BGR2Lab)  # transform from BGR to LAB
            self.last_frame_lightness = img_lab[:, :, 0].astype(np.int32)
            return False, []

        self.height, self.width, _ = img.shape

        # STEP 1: preprocessing

        img_mor, img_bw, img_sub_norm, self.last_frame_lightness = preprocess(self.last_frame_lightness, img,
                                                                                   self.reverse_flag)

        # STEP 2: find corners
        tag_corners_list = find_corner_using_contours(img_mor)

        # # ======= to display =========
        # img_poly_lines = img_sub_norm.copy()
        # for tagCorners in tag_corners_list:
        #     cv2.polylines(img_poly_lines, [tagCorners], True, 255)
        # cv2.imshow("imgWithPoints", img_poly_lines)

        # STEP 3 : decoding
        result_list = decode(img_bw, tag_corners_list, self.tag36h11_info)

        # ======= to display =========
        imgFinal = img_sub_norm.copy()
        for tagCorners in tag_corners_list:
            cv2.polylines(imgFinal, [tagCorners], True, 255)
        for result in result_list:
            text = "idx:" + str(result["idx"]) + " hamming:" + str(result["hamming"])
            org = (result["lt_rt_rd_ld"][0, :].item(0), result["lt_rt_rd_ld"][0, :].item(1))
            cv2.putText(imgFinal, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)

        cv2.imshow("imgWithResults", imgFinal)
        cv2.waitKey(0)

        # STEP 4 : update the flag
        if len(result_list) == 0:
            self.tag_flag = False
        else:
            self.tag_flag = True
            self.reverse_flag = -1 * self.reverse_flag  # 下一张是反转的检测的
            self.pass_flag = - self.pass_flag  # 去除下一张

        return self.tag_flag, result_list




def imshow_img(name, img):
    '''
    show an image whatever its data format is
    :param name: str
    :param img: np.array
    :return: None
    '''
    cv2.imshow(name, img.astype(np.uint8))
    cv2.waitKey(0)


def imshow_corners():
    pass


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
