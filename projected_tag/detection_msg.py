#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: detection_msg.py
Date: 2021/6/4 10:18
LastEditors: LI Jinjie
LastEditTime: 2021/6/4 10:18
Description: file content
'''


class Detection:
    def __init__(self, tag_family, tag_id, hamming, corners):
        self.tag_family = tag_family
        self.tag_id = tag_id
        self.hamming = hamming
        self.corners = corners
