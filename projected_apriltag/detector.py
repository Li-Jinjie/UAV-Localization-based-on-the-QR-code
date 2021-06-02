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
from .preprocessing import get_lightness_ch, preprocess
from .finding_corners import find_corner_using_contours
from .decoding import decode
from .pose_estimation import PoseEstimator


class TagsDetector:

    def __init__(self, path_map, path_calibration_para=None):
        # to correct distortion
        self.cam_mtx = None
        self.cam_dist = None
        self.cam_new_mtx = None
        self.roi = None
        if path_calibration_para is not None:
            para = np.load(path_calibration_para)
            self.cam_mtx = para['mtx']
            self.cam_dist = para['dist']

        # to decode
        self.tag36h11_info = Tag36H11()

        # to estimate pose
        self.pose_estimator = PoseEstimator(path_map)

        # last frame and frame information
        self.f_lightness_pre = None

        # flags
        self.refine_cam_mtx_flag = False
        self.tag_exist_flag = True
        self.reverse_flag = -1
        self.pass_flag = -1

    def detect(self, img):
        """
        Detect the apriltags in the image.
        :param img:  BGR
        :return self.tag_flag, result_list:
        """
        # STEP 0: correcting distortion
        if self.cam_dist is not None:
            if self.refine_cam_mtx_flag is False:
                h, w = img.shape[0:2]
                self.cam_new_mtx, self.roi = cv2.getOptimalNewCameraMatrix(self.cam_mtx, self.cam_dist, (w, h), 1,
                                                                           (w, h))
                self.refine_cam_mtx_flag = True
            img_undist = cv2.undistort(img, self.cam_mtx, self.cam_dist, None, self.cam_new_mtx)
            # cv2.imshow('img', img)
            # cv2.imshow('img_undist', img_undist)
            # cv2.waitKey(0)
            img = img_undist

        # decide if pass and save the frame or not
        if self.pass_flag > 0:
            self.pass_flag = - self.pass_flag
            f_lightness_now = get_lightness_ch(img, self.roi)
            self.f_lightness_pre = f_lightness_now
            return False, []

        # STEP 1: format conversion and preprocessing
        # 1) convert BGR to Lab, get lightness channel, remove
        f_lightness_now = get_lightness_ch(img, self.roi)

        # 2) preprocessing, including alignment, normalization, smoothing, threshold and morphology
        img_mor, img_bw, img_sub_norm = preprocess(self.f_lightness_pre, f_lightness_now, self.reverse_flag)
        self.f_lightness_pre = f_lightness_now

        # STEP 2: find corners
        tag_corners_list = find_corner_using_contours(img_mor)

        # # ======= to display =========
        # img_poly_lines = img_sub_norm.copy()
        # for tagCorners in tag_corners_list:
        #     cv2.polylines(img_poly_lines, [tagCorners], True, 255)
        # cv2.imshow("imgWithPoints", img_poly_lines)

        # STEP 3: decoding
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

        # STEP 4: update the flag
        if len(result_list) == 0:
            self.tag_exist_flag = False
        else:
            self.tag_exist_flag = True
            self.reverse_flag = -1 * self.reverse_flag  # 下一张是反转的检测的
            self.pass_flag = - self.pass_flag  # 去除下一张

        # STEP 5: pose estimation
        if self.tag_exist_flag is True:
            pose = self.pose_estimator.pose_estimate(result_list)
            print(pose)

        return self.tag_exist_flag, result_list


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
    pass
#     # img = cv2.imread("receiver_pictures/L3_add_1.png", flags=cv2.IMREAD_GRAYSCALE)
#     img = cv2.imread("receiver_pictures/0517_bw_120fps_90degree_720p.avi_lab_2.png", flags=cv2.IMREAD_GRAYSCALE)
#     detector = TagsDetector()
#     flag, results = detector.detect(img)
#
#     if flag == True:
#         for i, result in enumerate(results):
#             print(result)
#         print(str(len(results)) + ' apriltags are detected in total!')
#     else:
#         print('No apriltag is detected!')
