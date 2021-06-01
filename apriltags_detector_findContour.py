#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: apriltags_detector_findContour.py
Date: 2021/5/7 8:07
LastEditors: LI Jinjie
LastEditTime: 2021/5/7 8:07
Description: a new AprilTag detector class
'''

import cv2
import numpy as np
import math
from tag36h11 import Tag36H11


class TagsDetector:

    def __init__(self):
        # to decode
        self.tag36h11_info = Tag36H11()

        # last frame and frame information
        self.last_frame_lightness = None
        self.height = None
        self.width = None

        # to save the results
        self.cornersList = []
        self.tagsList = []
        self.resultList = []  # id, hamming_distance, rotate_degree

        # flags
        self.tag_flag = True
        self.reverse_flag = -1
        self.pass_flag = 1

    def detect(self, img):
        """
        Detect the apriltags in the image.
        :param img:  BGR
        :return self.tag_flag, result_list:
        """
        # if self.pass_flag > 0:
        #     self.pass_flag = - self.pass_flag
        #     return False, []

        self.height, self.width, _ = img.shape

        # STEP 1: preprocessing
        # 1. convert BGR to Lab, alignment, subtraction, normalization
        imgLab = cv2.cvtColor(img, code=cv2.COLOR_BGR2Lab)  # transform from BGR to LAB
        img_lightness = imgLab[:, :, 0].astype(np.int32)

        if self.last_frame_lightness is None:
            img_subtraction = img_lightness
        else:
            frame_now_aligned = self._align_frames(self.last_frame_lightness, img_lightness)
            img_subtraction = frame_now_aligned - self.last_frame_lightness
        self.last_frame_lightness = img_lightness  # store frame
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
        if self.reverse_flag == 1:
            img_bw = 255 - img_bw
        # cv2.imshow("imgBW", img_bw)

        img_bw = cv2.medianBlur(img_bw, 5)
        # cv2.imshow("median_blur", img_bw)

        #  Morphology open, to remove some noise.
        kernel = np.ones((6, 6), np.uint8)
        img_mor = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("open", img_mor)

        kernel = np.ones((16, 16), np.uint8)  # need to adjust more carefully
        img_mor = cv2.morphologyEx(img_mor, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("close", img_mor)
        # cv2.waitKey(0)

        # STEP 2: find corners
        # 1. find contours
        contours, hierarchy = cv2.findContours(img_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # ====== to display ========
        # img_with_contours = img_sub_norm.copy()
        # img_with_contours = cv2.drawContours(img_with_contours, contours, -1, 255, 2)
        # cv2.imshow("img_with_contours", img_with_contours)

        # 2. get convex hull
        hull_list = []
        for contour in contours:
            hull = cv2.convexHull(contour, returnPoints=True)
            hull_list.append(hull)

        # # ====== to display ========
        # img_with_hull = img_sub_norm.copy()
        # for corners in hull_list:
        #     for i in range(corners.shape[0]):
        #         cv2.circle(img_with_hull, (corners[i, 0, :].item(0), corners[i, 0, :].item(1)), 2, 255, -1)
        # cv2.imshow("img_with_convex_hull_points", img_with_hull)

        # 3. contour approximation
        corners_list = []
        for c in hull_list:
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                if 0.8 <= ar <= 1.2 and w * h > 1000:
                    corners_list.append(approx)

        # # ====== to display ========
        # img_with_pts = img_sub_norm.copy()
        # for corners in corners_list:
        #     for i in range(corners.shape[0]):
        #         cv2.circle(img_with_pts, (corners[i, 0, :].item(0), corners[i, 0, :].item(1)), 2, 255, -1)
        # cv2.imshow("imgWithPoints", img_with_pts)
        # cv2.waitKey(0)

        # # (1) 得到每个四边形外接矩形的lt和rb坐标
        # rectangleList = []
        # for corners in corners_list:
        #     # cv2.fillConvexPoly(img, corners, 255)
        #     # cv2.fillPoly(black, [corners], 255)
        #
        #     lt = np.array([np.min(corners[:, 0, 0]), np.min(corners[:, 0, 1])])
        #     rb = np.array([np.max(corners[:, 0, 0]), np.max(corners[:, 0, 1])])
        #     rectangleList.append(np.array([lt, rb]))
        #
        #     # ====== to display ========
        #     for i in range(corners.shape[0]):
        #         cv2.circle(img, (corners[i, 0, :].item(0), corners[i, 0, :].item(1)), 2, 255, -1)
        # # cv2.imshow("imgWithCorners", img)
        # # cv2.waitKey(0)
        #
        # # (2) 判断矩形的重叠
        # N = len(rectangleList)
        # i = 0
        # while i != N:
        #     for j in range(i + 1, N):
        #         rect1 = rectangleList[i]  # 0: lt, 1: rb
        #         rect2 = rectangleList[j]
        #
        #         # 求相交位置的坐标
        #         p1 = np.max([rect1[0, :], rect2[0, :]], axis=0)  # lt
        #         p2 = np.min([rect1[1, :], rect2[1, :]], axis=0)  # rb
        #
        #         if (p2[0] > p1[0]) and (p2[1] > p1[1]):
        #             # add new rectangle
        #             lt_new = np.min([rect1[0, :], rect2[0, :]], axis=0)
        #             rb_new = np.max([rect1[1, :], rect2[1, :]], axis=0)
        #
        #             rectangleList.append(np.array([lt_new, rb_new]))
        #             # delete old
        #             rectangleList.pop(j)  # 先删除后边的元素
        #             rectangleList.pop(i)
        #             i = 0
        #             N = N - 1
        #             break
        #         else:
        #             j = j + 1
        #     i = i + 1

        # # ======= to display =========
        # imgPoints = img.copy()
        # for rect in rectangleList:
        #     cv2.rectangle(imgPoints, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), 255)
        # cv2.imshow("imgWithRectangles", imgPoints)
        # cv2.waitKey(0)

        # # (3) 得到矩形四个点的坐标
        # tag_corners_list = []  # 4 * 2
        # for rect in rectangleList:  # each rectangle
        #     tagCorners = np.zeros([4, 1, 2], dtype=np.int32)
        #     lt = rect[0]
        #     rb = rect[1]
        #     rt = np.array([rb.item(0), lt.item(1)])
        #     lb = np.array([lt.item(0), rb.item(1)])
        #     rect_corners = [lt, rt, rb, lb]
        #
        #     for idx, pts in enumerate(rect_corners):  # each point of a rectangle
        #         distance_min = 9999
        #         for corners in corners_list:
        #             # if in the rectangle
        #             if (lt.item(0) <= corners[0, 0, 0] <= rb.item(0)) and (
        #                     lt.item(1) <= corners[0, 0, 1] <= rb.item(1)):
        #                 for corner in corners:
        #                     distance = np.sum(np.abs(pts - corner))  # Manhattan Distance
        #                     if distance < distance_min:
        #                         distance_min = distance
        #                         tagCorners[idx, 0, :] = corner
        #             else:
        #                 continue
        #
        #     tag_corners_list.append(tagCorners)

        tag_corners_list = corners_list
        # # ======= to display =========
        # img_poly_lines = img_sub_norm.copy()
        # for tagCorners in tag_corners_list:
        #     cv2.polylines(img_poly_lines, [tagCorners], True, 255)
        # cv2.imshow("imgWithPoints", img_poly_lines)

        # STEP 3 : decoding
        result_list = self.decode(img_bw, tag_corners_list, self.tag36h11_info)

        # ======= to display =========
        imgFinal = img_sub_norm.copy()
        for tagCorners in tag_corners_list:
            cv2.polylines(imgFinal, [tagCorners], True, 255)
        for result in result_list:
            text = "idx:" + str(result["idx"]) + " hamming:" + str(result["hamming"])
            org = (result["lt_rt_rd_ld"][0, :].item(0), result["lt_rt_rd_ld"][0, :].item(1))
            cv2.putText(imgFinal, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)

        cv2.imshow("imgWithResults", imgFinal)
        cv2.waitKey(0)

        # STEP 4 : update the flag
        if len(result_list) == 0:
            self.tag_flag = False
        else:
            self.tag_flag = True
            self.reverse_flag = -1 * self.reverse_flag  # 下一张是反转的检测的
            self.pass_flag = - self.pass_flag  # 去除下一张

        return self.tag_flag, result_list

    def _align_frames(self, frame_pre, frame_now):
        """
        choose 3 horizontal lines and 3 vertical lines to get the best x,y bias, return aligned frame_now
        :param frame_pre:
        :param frame_now:
        :return frame_now_aligned:
        """
        height = self.height
        width = self.width
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
        for offset in range(-5, 5 + 1, 1):
            a_x = horizontal_lines[0][:, max(offset, 0):-1 + min(offset, 0)]
            b_x = horizontal_lines[1][:, max(-offset, 0):-1 + min(-offset, 0)]
            val_x = np.max(np.linalg.norm(a_x - b_x, 1, axis=1))  # norm 1
            # print('offset_x is:', offset, ' norm=', val_x)
            if val_x < min_x_val:
                min_x_val = val_x
                offset_x = offset

            a_y = vertical_lines[0][:, max(offset, 0):-1 + min(offset, 0)]
            b_y = vertical_lines[1][:, max(-offset, 0):-1 + min(-offset, 0)]
            val_y = np.max(np.linalg.norm(a_y - b_y, 1, axis=1))  # norm 1
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

    def decode(self, img, tag_corners_list, tag36h11):
        """
        input the positions of the tags' four corners, then decode the tag
        :param img: image
        :param tag_corners_list: a list that contains the four corners' positions of all tags
        :param tag36h11: a class that stores tag36h11 information
        :return result_list: a list that contains 'idx, hamming_distance, rotate_degree', should be refreshed every time
        """
        result_list = []
        # a) Perspective transform
        tags_list = self._perspective_transform(img, tag_corners_list)
        # b) save data of tag36h11 in __init__()
        for i, tag in enumerate(tags_list):
            # c) rotate the img and get four code
            int_code_list = self._rotate_get_int(tag, tag36h11.bit_x, tag36h11.bit_y)
            # d) calculate the hamming distance and find the minimal one
            hamming, idx, rotate_degree = self._find_min_hamming(int_code_list, tag36h11.codes)
            # e) filter invalid code and append result
            if hamming < 4:
                lt_rt_rd_ld = np.rot90(tag_corners_list[i], int(rotate_degree / 90))
                result_list.append({'idx': idx, 'hamming': hamming, 'lt_rt_rd_ld': lt_rt_rd_ld})

        return result_list

    def _perspective_transform(self, img_mor, corners_list):
        tags_list = []
        size_pixel = 80
        points2 = np.array([[0, 0], [size_pixel, 0], [size_pixel, size_pixel], [0, size_pixel]])  # lt_rt_rd_ld
        for corners in corners_list:
            # cv2.polylines(img_mor, [corners], True, 127)
            # cv2.imshow("img_mor", img_mor)
            # cv2.waitKey(0)

            corners = corners.squeeze()
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

    def _rotate_get_int(self, tag, bit_x, bit_y):
        int_code_list = []
        for i in range(4):
            tag = np.rot90(tag, i)
            code = '0b'
            for j in range(36):
                if tag[bit_y[j], bit_x[j]] < 100:
                    code = code + '0'
                else:
                    code = code + '1'
            int_code = int(code, 2)
            int_code_list.append(int_code)
        return int_code_list

    def _find_min_hamming(self, int_code_list, tag36h11_list):
        hamming_min = 36
        id_min = 0
        rotate_degree = 0
        for int_code in int_code_list:
            hamming_min_local = 36
            id_min_local = 0
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
            if hamming_min == 0:
                break
            rotate_degree += 90

        return hamming_min, id_min, rotate_degree


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
    # img = cv2.imread("receiver_pictures/L3_add_1.png", flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("receiver_pictures/0517_bw_120fps_90degree_720p.avi_lab_2.png", flags=cv2.IMREAD_GRAYSCALE)
    detector = TagsDetector()
    flag, results = detector.detect(img)

    if flag == True:
        for i, result in enumerate(results):
            print(result)
        print(str(len(results)) + ' apriltags are detected in total!')
    else:
        print('No apriltag is detected!')
