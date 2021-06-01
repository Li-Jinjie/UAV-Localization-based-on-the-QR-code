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
    def __init__(self):
        path = "D:\\ForGithub\\UAV-Localization-based-on-the-QR-code\\apriltag_map\\maps_info.yaml"
        with open(path, 'r') as file:
            map_info = yaml.load(file, Loader=yaml.FullLoader)
            self.unit = map_info['unit']
            self.width_num = map_info['width_num']
            self.height_num = map_info['height_num']
            self.layout = map_info['layout']

    def pose_estimate(self):
        pass

