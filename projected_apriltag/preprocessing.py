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


def get_lightness_ch(img, roi):
    img_lab = cv2.cvtColor(img, code=cv2.COLOR_BGR2Lab)  # transform from BGR to LAB
    f_lightness_now = img_lab[:, :, 0].astype(np.int32)
    if roi is not None:
        h_org, w_org = img.shape[0:2]
        mask = np.ones((h_org, w_org), dtype=np.bool)
        x, y, w, h = roi
        mask[y:y + h, x:x + w] = False
        # cv2.imshow('f_lightness_now', f_lightness_now.astype(np.uint8))
        f_lightness_now[mask] = 127
        # cv2.imshow('f_lightness_now_masked', f_lightness_now.astype(np.uint8))
        # cv2.waitKey(0)

    return f_lightness_now


def img_preprocess(f_lightness_pre, f_lightness_now, reverse_flag):
    # 1) align image and subtraction
    if f_lightness_pre is None:
        img_subtraction = f_lightness_now
    else:
        frame_now_aligned, offset_x, offset_y = _align_frames(f_lightness_pre, f_lightness_now)
        img_subtraction = frame_now_aligned - f_lightness_pre
        # remove the outer black border
        if offset_x != 0:
            img_subtraction[:, 0:np.abs(offset_x)] = 0
            img_subtraction[:, -np.abs(offset_x):] = 0
        if offset_y != 0:
            img_subtraction[0:np.abs(offset_y), :] = 0
            img_subtraction[-np.abs(offset_y):, :] = 0

    # 2) normalization
    img_sub_norm = ((img_subtraction - np.min(img_subtraction)) * 255 / (
            np.max(img_subtraction) - np.min(img_subtraction))).astype(np.uint8)

    # 3) smoothing and threshold
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

    # 4) morphology
    #  Morphology open, to remove some noise.
    kernel = np.ones((6, 6), np.uint8)
    img_mor = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("open", img_mor)

    kernel = np.ones((15, 15), np.uint8)  # need to adjust more carefully
    img_mor = cv2.morphologyEx(img_mor, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("close", img_mor)
    # cv2.waitKey(0)
    return img_mor, img_bw, img_sub_norm


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
    for offset in range(-5, 5 + 1, 1):   # TODO: check this, [-7, 7]?
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

    return frame_now_aligned, offset_x, offset_y
