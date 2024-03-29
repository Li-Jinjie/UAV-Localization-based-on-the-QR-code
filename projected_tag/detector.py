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
from pupil_apriltags import Detector
from .tag36h11 import Tag36H11
from .preprocessing import get_lightness_ch, img_preprocess
from .finding_corners import find_corner_using_contours
from .decoding import decode
from .pose_estimation import PoseEstimator
from .detection_msg import Detection


class ProjectedTagsDetector:

    def __init__(self, path_map, path_calibration_para=None, detector_type='AprilTag3'):
        # detector_type: 'AprilTag3', 'ByMe', 'ArUcoOpenCV'
        # estimate_method: 'average' or 'all_pts'
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
        self._tag36h11_info = Tag36H11()

        # to estimate pose
        self._pose_estimator = PoseEstimator(path_map, self.cam_mtx, self.cam_dist)

        # last frame and frame information
        self.f_lightness_pre = None

        # flags
        self.detector_type = detector_type
        if self.detector_type == 'AprilTag3':
            self.detector = Detector(families='tag36h11',
                                     nthreads=1,
                                     quad_decimate=2.0,
                                     quad_sigma=0.0,
                                     refine_edges=1,
                                     decode_sharpening=0,
                                     debug=0)
        elif self.detector_type == 'ArUcoOpenCV':
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.refine_cam_mtx_flag = False
        self.tag_exist_flag = True
        self.reverse_flag = -1
        self.pass_flag = -1

    def detect(self, img):
        """
        Detect the apriltags in the image.
        :param img:  BGR
        :return self.tag_flag, results, xyz:
        """
        # STEP 0: correcting distortion
        # if self.cam_dist is not None:
        #     if self.refine_cam_mtx_flag is False:
        #         h, w = img.shape[0:2]
        #         # cam_new_mtx is only used to undistort the image.
        #         self.cam_new_mtx, self.roi = cv2.getOptimalNewCameraMatrix(self.cam_mtx, self.cam_dist, (w, h), 1,
        #                                                                    (w, h))
        #         self.refine_cam_mtx_flag = True
        #     img_undist = cv2.undistort(img, self.cam_mtx, self.cam_dist, None, self.cam_new_mtx)
        #     # cv2.imshow('img', img)
        #     # cv2.imshow('img_undist', img_undist)
        #     # cv2.waitKey(0)
        #     img = img_undist

        # decide if pass and save the frame or not
        if self.pass_flag > 0:
            self.pass_flag = - self.pass_flag
            f_lightness_now = get_lightness_ch(img, self.roi)
            self.f_lightness_pre = f_lightness_now
            return False, []

        # STEP 1: format conversion and preprocessing
        # 1) convert BGR to Lab, get lightness channel, remove black borders caused by undistortion
        f_lightness_now = get_lightness_ch(img, self.roi)

        # 2) img preprocessing, including alignment, normalization, smoothing, threshold and morphology
        img_mor, img_bw, img_sub_norm = img_preprocess(self.f_lightness_pre, f_lightness_now, self.reverse_flag)

        # # TODO: delete the following code when execution, because they are used for display the effect of shake elimination
        # f_lightness_no_align = get_lightness_ch(img)
        # if self.f_lightness_pre is not None:
        #     img_sub_no_align = f_lightness_no_align - self.f_lightness_pre
        #     img_sub_no_align = ((img_sub_no_align - np.min(img_sub_no_align)) * 255 / (
        #             np.max(img_sub_no_align) - np.min(img_sub_no_align))).astype(np.uint8)
        # else:
        #     img_sub_no_align = f_lightness_no_align
        # #######################################################

        self.f_lightness_pre = f_lightness_now

        if self.detector_type == 'AprilTag3':
            # official one
            results = self.detector.detect(img_mor, estimate_tag_pose=False)
        elif self.detector_type == 'ByMe':
            # my implementation
            # STEP 2: find corners
            tag_corners_list = find_corner_using_contours(img_mor)
            # STEP 3: decoding
            results = decode(img_bw, tag_corners_list, self._tag36h11_info)
        elif self.detector_type == 'ArUcoOpenCV':
            (corners_set, ids, _) = cv2.aruco.detectMarkers(img_mor, self.aruco_dict, parameters=self.aruco_params)
            # The order of the corners in their ORIGINAL order (which is clockwise starting with top left), no rotation
            # corners should be adjusted in the counter-clockwise order: ld_rd_rt_lt, [4, 2]

            results = []
            for i in range(len(corners_set)):
                results.append(Detection(b'tag36h11', ids[i].item(), -1, corners_set[i][0, :, :]))
                print(corners_set[i][:, ::-1, :])

        # STEP 4: update the flag
        if len(results) == 0:
            self.tag_exist_flag = False
        else:
            self.tag_exist_flag = True
            self.reverse_flag = -1 * self.reverse_flag  # 下一张是反转的检测的
            self.pass_flag = - self.pass_flag  # 去除下一张

        # # ======= to display =========
        # if self.tag_exist_flag is True:
        #     cv2.imshow("img_org", img)
        #     cv2.imshow("shake_elimination", img_sub_norm)
        #     cv2.imshow("img_preprocessing", img_mor)
        #
        #     img_quads_detect = img_sub_norm.copy()
        #     for result in results:
        #         cv2.polylines(img_quads_detect, [result.corners.astype(np.int32)], True, 255)
        #         # text = 'tag_id:' + str(result.tag_id) + ' hamming:' + str(result.hamming)
        #         # org = (result.corners[-1, 0].astype(np.int32), result.corners[-1, 1].astype(np.int32))
        #         # cv2.putText(img_quads_detect, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
        #         for corner in result.corners.astype(np.int32):
        #             img_quads_detect = cv2.circle(img_quads_detect, tuple(corner), 6, 255, -1)
        #     cv2.imshow("quads detection", img_quads_detect)
        #
        # # --------- final results ---------
        #     img_final_3 = np.zeros_like(img)
        #     img_final_3[:, :, 0] = img_sub_norm
        #     img_final_3[:, :, 1] = img_sub_norm
        #     img_final_3[:, :, 2] = img_sub_norm
        #
        #     for result in results:
        #         corners = result.corners.astype(np.int32)
        #         cv2.polylines(img_final_3, [corners], True, (255, 0, 0), 3)
        #         img_final_3 = cv2.line(img_final_3, tuple(corners[0, :].ravel()),
        #                                tuple(corners[1, :].ravel()), (0, 0, 255), 3)  # red for x axis
        #         img_final_3 = cv2.line(img_final_3, tuple(corners[0, :].ravel()),
        #                                tuple(corners[-1, :].ravel()), (0, 255, 0), 3)  # green for y axis
        #         # text = 'tag_id:' + str(result.tag_id) + ' hamming:' + str(result.hamming)
        #         text = str(result.tag_id)
        #         org = (np.mean(result.corners[:, 0]).astype(np.int32) - 40,
        #                np.mean(result.corners[:, 1]).astype(np.int32) + 30)
        #         cv2.putText(img_final_3, text, org, cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
        #     cv2.imshow("final", img_final_3)
        #
        #     cv2.waitKey(0)
        #     # ---- save images ------
        #     cv2.imwrite("1_img_org.png", img)
        #     cv2.imwrite("2_img_no_alignment.png", img_sub_no_align)
        #     cv2.imwrite("3_shake_elimination.png", img_sub_norm)
        #     cv2.imwrite("4_img_preprocessing.png", img_mor)
        #     # cv2.imwrite("quads_detection.png", img_quads_detect)
        #     cv2.imwrite("5_final_result.png", img_final_3)

        return self.tag_exist_flag, results

    def estimate_pose(self, tag_exist_flag, result_list, estimate_method='average', ransac_flag=True):
        """
        estimate pose
        :param tag_exist_flag: if exist tags?
        :param result_list: detection result
        :param estimate_method: 'average' or 'all_pts'
        :param ransac_flag: use SolvePnPRansac() or not? only useful when choose 'all_pts' methods
        :return rvec, tvec: [position and rotation vector] or [None, None]
        """
        # pose estimation and validate the data
        if tag_exist_flag is True:
            rvec, tvec = self._pose_estimator.estimate_pose(result_list, estimate_method=estimate_method,
                                                            ransac_flag=ransac_flag)
            return rvec, tvec
        return None, None


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
