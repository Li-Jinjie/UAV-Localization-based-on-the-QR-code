#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: map_info_loading.py
Date: 2021/6/1 22:35
LastEditors: LI Jinjie
LastEditTime: 2021/6/1 22:35
Description: load map information
'''
import yaml

path = "D:\\ForGithub\\UAV-Localization-based-on-the-QR-code\\apriltag_map\\maps_info.yaml"
with open(path, 'r') as file:
    map_info = yaml.load(file, Loader=yaml.FullLoader)
    layout = map_info['layout']
    print(layout[0]['size'])
