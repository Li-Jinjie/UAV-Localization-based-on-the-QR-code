#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: preprocessing.py
Date: 2021/6/1 20:01
LastEditors: LI Jinjie
LastEditTime: 2021/6/1 20:01
Description: file content
'''
import numpy as np
import cv2

def preprocess(last_frame_lightness, img, reverse_flag):
    # 1). convert BGR to Lab, alignment, subtraction, normalization
    img_lab = cv2.cvtColor(img, code=cv2.COLOR_BGR2Lab)  # transform from BGR to LAB
    img_lightness = img_lab[:, :, 0].astype(np.int32)

    if last_frame_lightness is None:
        img_subtraction = img_lightness
    else:
        frame_now_aligned = _align_frames(last_frame_lightness, img_lightness)
        img_subtraction = frame_now_aligned - last_frame_lightness
    img_sub_norm = ((img_subtraction - np.min(img_subtraction)) * 255 / (
            np.max(img_subtraction) - np.min(img_subtraction))).astype(np.uint8)

    # 2. smoothing, threshold and morphology
    median_value = np.median(img_sub_norm)
    img_smooth = cv2.blur(img_sub_norm, (5, 5))
    _, img_bw = cv2.threshold(img_smooth, median_value, 255, cv2.THRESH_BINARY)  # extremely critical
    # cv2.imshow("org", img_sub_norm)
    # cv2.imshow("smooth", img_smooth)
    # cv2.imshow("threshold", img_bw)
    # _, img_bw_org = cv2.threshold(img_sub_norm, median_value, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold_org", img_bw_org)

    # _, img_bw = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)  # extremely critical
    if reverse_flag == 1:
        img_bw = 255 - img_bw
    # cv2.imshow("imgBW", img_bw)

    img_bw = cv2.medianBlur(img_bw, 5)
    # cv2.imshow("median_blur", img_bw)

    #  Morphology open, to remove some noise.
    kernel = np.ones((6, 6), np.uint8)
    img_mor = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("open", img_mor)

    kernel = np.ones((15, 15), np.uint8)  # need to adjust more carefully
    img_mor = cv2.morphologyEx(img_mor, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("close", img_mor)
    # cv2.waitKey(0)
    return img_mor, img_bw, img_sub_norm, img_lightness


def _align_frames(frame_pre, frame_now):
    """
    choose 3 horizontal lines and 3 vertical lines to get the best x,y bias, return aligned frame_now
    :param frame_pre:
    :param frame_now:
    :return frame_now_aligned:
    """
    height, width = frame_pre.shape
    horizontal_lines = [np.array([frame_pre[int(height * 1 / 4), :],
                                  frame_pre[int(height * 2 / 4), :],
                                  frame_pre[int(height * 3 / 4), :]]),

                        np.array([frame_now[int(height * 1 / 4), :],
                                  frame_now[int(height * 2 / 4), :],
                                  frame_now[int(height * 3 / 4), :]])]

    vertical_lines = [np.array([frame_pre[:, int(width * 1 / 4)],
                                frame_pre[:, int(width * 2 / 4)],
                                frame_pre[:, int(width * 3 / 4)]]),

                      np.array([frame_now[:, int(width * 1 / 4)],
                                frame_now[:, int(width * 2 / 4)],
                                frame_now[:, int(width * 3 / 4)]])]

    min_x_val = 99999
    min_y_val = 99999
    offset_x = 0
    offset_y = 0
    for offset in range(-7, 7 + 1, 1):
        a_x = horizontal_lines[0][:, max(offset, 0):-1 + min(offset, 0)]
        b_x = horizontal_lines[1][:, max(-offset, 0):-1 + min(-offset, 0)]
        val_x = np.mean(np.linalg.norm(a_x - b_x, 1, axis=1))  # norm 1
        # print('offset_x is:', offset, ' norm=', val_x)
        if val_x < min_x_val:
            min_x_val = val_x
            offset_x = offset

        a_y = vertical_lines[0][:, max(offset, 0):-1 + min(offset, 0)]
        b_y = vertical_lines[1][:, max(-offset, 0):-1 + min(-offset, 0)]
        val_y = np.mean(np.linalg.norm(a_y - b_y, 1, axis=1))  # norm 1
        # print('offset_y is:', offset, ' norm=', val_y)
        if val_y < min_y_val:
            min_y_val = val_y
            offset_y = offset

    print('The final offset_x is:', offset_x)
    print('The final offset_y is:', offset_y)

    # translation
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    frame_now_aligned = cv2.warpAffine(frame_now.astype(np.uint8), M, (width, height)).astype(np.int32)

    return frame_now_aligned