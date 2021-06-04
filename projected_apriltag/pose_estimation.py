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

    def estimate_pose(self, result_list):
        obj_pts_set = np.zeros([1, len(result_list) * 4, 3])  # 3D points   size: 1*N *3
        img_pts_set = np.zeros([1, len(result_list) * 4, 2])  # 2D image points   size: 1*N * 2
        for i, result in enumerate(result_list):
            idx = result.tag_id
            obj_pts = self.lookup_map(idx)  # 3D points   size: 4*3
            obj_pts *= 1000  # from meter to mm
            obj_pts_set[:, 4 * i:4 * (i + 1), :] = obj_pts
            # 2D image points   size: 4*2  counter-clock:ld_rd_rt_lt
            img_pts = result.corners.astype(np.float64)
            img_pts = img_pts[::-1, :]  # from counter-clockwise to clockwise to meet IPPE_SQUARE's requirement
            img_pts_set[:, 4 * i:4 * (i + 1), :] = img_pts

        ret, rvec, tvec = cv2.solvePnP(obj_pts_set, img_pts_set, self.cam_mtx, self.cam_dist,
                                       flags=cv2.SOLVEPNP_IPPE)
        if ret is True:
            print('solvePnP() is True!')
            return rvec, tvec
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
