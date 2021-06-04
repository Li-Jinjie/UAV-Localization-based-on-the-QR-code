#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: decoding.py
Date: 2021/6/1 19:55
LastEditors: LI Jinjie
LastEditTime: 2021/6/1 19:55
Description: decoding apriltags
'''
import numpy as np
import cv2
from .detection_msg import Detection


def decode(img, tag_corners_list, tag36h11):
    """
    input the positions of the tags' four corners, then decode the tag
    :param img: image
    :param tag_corners_list: a list that contains the four corners' positions of all tags
    :param tag36h11: a class that stores tag36h11 information
    :return results: a list that contains 'idx, hamming_distance, rotate_degree', should be refreshed every time
    """
    results = []
    # a) Perspective transform
    tags_list = _perspective_transform(img, tag_corners_list)
    # b) save data of tag36h11 in __init__()
    for i, tag in enumerate(tags_list):
        # c) counter-clockwise rotate the img and get four code
        int_code_list = _rotate_get_int(tag, tag36h11.bit_x, tag36h11.bit_y)
        # d) calculate the hamming distance and find the minimal one
        hamming, idx, rotate_degree = _find_min_hamming(int_code_list, tag36h11.codes)
        # e) filter invalid code and append result
        if hamming < 3:
            # -1:counter-clockwise, +1: clockwise
            ld_rd_rt_lt = np.roll(tag_corners_list[i], -int(rotate_degree / 90), axis=0)
            results.append(Detection(b'tag36h11', idx, hamming, ld_rd_rt_lt))

    return results


def _perspective_transform(img_mor, corners_list):
    tags_list = []
    size_pixel = 80
    # points2 = np.array([[0, 0], [size_pixel, 0], [size_pixel, size_pixel], [0, size_pixel]])  # lt_rt_rd_ld
    points2 = np.array([[0, size_pixel], [size_pixel, size_pixel], [size_pixel, 0], [0, 0]])  # ld_rd_rt_lt
    for corners in corners_list:
        # cv2.polylines(img_mor, [corners], True, 127)
        # cv2.imshow("img_mor", img_mor)
        # cv2.waitKey(0)

        h, status = cv2.findHomography(corners, points2)
        qr_code = cv2.warpPerspective(img_mor, h, (size_pixel, size_pixel))

        qr_blur = cv2.medianBlur(qr_code, 3)  # important!

        qr_code_small = cv2.resize(qr_blur, (8, 8), dst=cv2.INTER_NEAREST)
        _, qr_code_small = cv2.threshold(qr_code_small, 0, 255, cv2.THRESH_OTSU)
        if (np.sum(qr_code_small[0, :-1]) + np.sum(qr_code_small[:-1, -1]) + np.sum(qr_code_small[1:, 0]) + np.sum(
                qr_code_small[-1, 1:])) / 28 > 127:
            qr_code_small = 255 - qr_code_small

        tags_list.append(qr_code_small)
    return tags_list


def _rotate_get_int(tag, bit_x, bit_y):
    int_code_list = []
    for i in range(4):
        tag_rot = np.rot90(tag, i)
        code = '0b'
        for j in range(36):
            if tag_rot[bit_y[j], bit_x[j]] < 127:
                code = code + '0'
            else:
                code = code + '1'
        int_code = int(code, 2)
        int_code_list.append(int_code)
    return int_code_list


def _find_min_hamming(int_code_list, tag36h11_list):
    hamming_min = 36
    id_min = 0
    rot_idx_min = 0
    # for different rotation angles
    for rot_idx, int_code in enumerate(int_code_list):
        hamming_min_local = 36
        id_min_local = 0

        # find the most likely tag id for this rotation angles
        for idx, tag_code in enumerate(tag36h11_list):
            s = str(bin(int_code ^ tag_code))
            hamming = 0
            for i in range(2, len(s)):
                if int(s[i]) == 1:
                    hamming += 1
            if hamming_min_local > hamming:
                hamming_min_local = hamming
                id_min_local = idx
            if hamming_min_local == 0:
                break

        if hamming_min > hamming_min_local:
            hamming_min = hamming_min_local
            id_min = id_min_local
            rot_idx_min = rot_idx
        if hamming_min == 0:
            break

    return hamming_min, id_min, 90 * rot_idx_min
