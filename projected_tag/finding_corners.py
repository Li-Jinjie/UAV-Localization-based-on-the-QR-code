#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: finding_corners.py
Date: 2021/6/1 19:58
LastEditors: LI Jinjie
LastEditTime: 2021/6/1 19:58
Description: file content
'''
import cv2
import numpy as np


def find_corner_using_contours(img_mor):
    # 1. find contours
    contours, hierarchy = cv2.findContours(img_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Outer contours should be oriented counter-clockwise, inner contours clockwise.

    # # ====== to display ========
    # img_with_contours = img_sub_norm.copy()
    # img_with_contours = cv2.drawContours(img_with_contours, contours, -1, 255, 2)
    # cv2.imshow("img_with_contours", img_with_contours)

    corners_list = []
    for contour in contours:
        # 2. get convex hull
        hull = cv2.convexHull(contour, returnPoints=True)
        # 3. contour approximation
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if 0.8 <= ar <= 1.2 and w * h > 1000:
                corners = np.squeeze(approx, axis=1)
                corners = corners[::-1, :]   # from clockwise to counter-clockwise
                corners_list.append(corners)

    # # ====== to display ========
    # img_with_pts = img_sub_norm.copy()
    # for corners in corners_list:
    #     for i in range(corners.shape[0]):
    #         cv2.circle(img_with_pts, (corners[i, 0, :].item(0), corners[i, 0, :].item(1)), 2, 255, -1)
    # cv2.imshow("imgWithPoints", img_with_pts)
    # cv2.waitKey(0)
    return corners_list
