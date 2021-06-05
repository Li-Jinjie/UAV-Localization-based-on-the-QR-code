#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: pose_estimation.py
Date: 2021/6/1 22:51
LastEditors: LI Jinjie
LastEditTime: 2021/6/1 22:51
Description: pose_estimation
'''
import numpy as np
import cv2
import yaml


class PoseEstimator:
    def __init__(self, path, cam_mtx, cam_dist):
        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist

        with open(path, 'r') as file:
            map_info = yaml.load(file, Loader=yaml.FullLoader)
            self.unit = map_info['unit']
            self.width_num = map_info['width_num']
            self.height_num = map_info['height_num']
            self.layout = map_info['layout']

    def estimate_pose(self, result_list, estimate_method='average', ransac_flag=True):
        # estimate_method: 'average', 'all_pts'
        num_result = len(result_list)
        if estimate_method == 'average':
            # Method 1: use IPPE_SQUARE to estimate every tags' pose, then average
            rvec_set = np.zeros([3, num_result])
            tvec_set = np.zeros([3, num_result])
            for i, result in enumerate(result_list):
                idx = result.tag_id
                obj_pts = self.lookup_map(idx)  # 3D points   size: 4*3
                obj_pts *= 1000  # from meter to mm
                obj_pts = np.expand_dims(obj_pts, axis=0)

                # 2D image points   size: 4*2  counter-clock:ld_rd_rt_lt
                img_pts = result.corners.astype(np.float64)
                img_pts = img_pts[::-1, :]  # from counter-clockwise to clockwise to meet IPPE_SQUARE's requirement
                img_pts = np.expand_dims(img_pts, axis=0)
                ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.cam_mtx, self.cam_dist,
                                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ret is True:
                    rvec_set[:, i:i + 1] = rvec
                    tvec_set[:, i:i + 1] = tvec

            # in case some ret is False
            valid_num = num_result - np.sum(
                np.prod([[tvec_set[0] == 0], [tvec_set[1] == 0], [tvec_set[2] == 0]], axis=2))
            return np.sum(rvec_set, axis=1) / valid_num, np.sum(tvec_set, axis=1) / valid_num

        elif estimate_method == 'all_pts':
            # Method 2: use IPPE to estimate all tags' pose at one time
            obj_pts_set = np.zeros([1, num_result * 4, 3])  # 3D points   size: 1*N *3
            img_pts_set = np.zeros([1, num_result * 4, 2])  # 2D image points   size: 1*N * 2
            for i, result in enumerate(result_list):
                idx = result.tag_id
                obj_pts = self.lookup_map(idx)  # 3D points   size: 4*3
                obj_pts *= 1000  # from meter to mm
                obj_pts_set[:, 4 * i:4 * (i + 1), :] = obj_pts
                # 2D image points   size: 4*2  counter-clock:ld_rd_rt_lt
                img_pts = result.corners.astype(np.float64)
                img_pts = img_pts[::-1, :]  # from counter-clockwise to clockwise to meet IPPE_SQUARE's requirement
                img_pts_set[:, 4 * i:4 * (i + 1), :] = img_pts
            if ransac_flag is True:
                ret, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts_set, img_pts_set, self.cam_mtx, self.cam_dist,
                                                              flags=cv2.SOLVEPNP_IPPE)
            else:
                ret, rvec, tvec = cv2.solvePnP(obj_pts_set, img_pts_set, self.cam_mtx, self.cam_dist,
                                               flags=cv2.SOLVEPNP_IPPE)

            if ret is True:
                print('solve PnP is True!')
                return rvec, tvec
        else:
            raise ValueError("the parameter estimate_method should only be 'average' or 'all_pts'")

        return None, None

    def lookup_map(self, idx):
        tag_dict = self.layout[idx]

        # The assert statement exists in almost every programming language. It helps detect problems early in your
        # program, where the cause is clear, rather than later when some other operation fails.
        assert idx == tag_dict['id'], "id number is wrong, please check carefully."

        bias = tag_dict['size'] / 2
        x = tag_dict['x']
        y = tag_dict['y']
        # this order is consistent with the required order of IPPE_SQUARE method.
        obj_pts = np.array([[x - bias, y + bias, 0],
                            [x + bias, y + bias, 0],
                            [x + bias, y - bias, 0],
                            [x - bias, y - bias, 0]])
        return obj_pts
