#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: test_img_movement.py
Date: 2021/5/26 20:02
LastEditors: LI Jinjie
LastEditTime: 2021/5/26 20:02
Description: file content
'''
import cv2
import numpy as np
from projected_apriltag.detector import TagsDetector

frame_1 = cv2.imread("data_real/receiver_pictures/1502.png")
frame_Lab_1 = cv2.cvtColor(frame_1, code=cv2.COLOR_BGR2Lab)  # transform from BGR to LAB
frame_intense_1 = frame_Lab_1[:, :, 0].astype(np.int32)

frame_2 = cv2.imread("data_real/receiver_pictures/1503.png")
frame_Lab_2 = cv2.cvtColor(frame_2, code=cv2.COLOR_BGR2Lab)  # transform from BGR to LAB
frame_intense_2 = frame_Lab_2[:, :, 0].astype(np.int32)

height, width = frame_intense_2.shape

# get best bias
# choose three lines
horizontal_lines = [np.array([frame_intense_1[int(height * 1 / 4), :],
                              frame_intense_1[int(height * 2 / 4), :],
                              frame_intense_1[int(height * 3 / 4), :]]),

                    np.array([frame_intense_2[int(height * 1 / 4), :],
                              frame_intense_2[int(height * 2 / 4), :],
                              frame_intense_2[int(height * 3 / 4), :]])]

vertical_lines = [np.array([frame_intense_1[:, int(width * 1 / 4)],
                            frame_intense_1[:, int(width * 2 / 4)],
                            frame_intense_1[:, int(width * 3 / 4)]]),

                  np.array([frame_intense_2[:, int(width * 1 / 4)],
                            frame_intense_2[:, int(width * 2 / 4)],
                            frame_intense_2[:, int(width * 3 / 4)]])]

# min_val = 99999
# offset_x = 0
# for offset in range(-5, 5 + 1, 1):
#     a = horizontal_lines[0][:, max(offset, 0):-1 + min(offset, 0)]
#     b = horizontal_lines[1][:, max(-offset, 0):-1 + min(-offset, 0)]
#     val = np.max(np.linalg.norm(a - b, 1, axis=1))  # norm 1
#     print('offset_x is:', offset, ' norm=', val)
#     if val < min_val:
#         min_val = val
#         offset_x = offset
#
# print('The final offset_x is:', offset_x)
#
# min_val = 99999
# offset_y = 0
# for offset in range(-5, 5 + 1, 1):
#     a = vertical_lines[0][:, max(offset, 0):-1 + min(offset, 0)]
#     b = vertical_lines[1][:, max(-offset, 0):-1 + min(-offset, 0)]
#     val = np.max(np.linalg.norm(a - b, 1, axis=1))  # norm 1
#     print('offset_y is:', offset, ' norm=', val)
#     if val < min_val:
#         min_val = val
#         offset_y = offset
#
# print('The final offset_y is:', offset_y)
min_x_val = 99999
min_y_val = 99999
offset_x = 0
offset_y = 0
for offset in range(-5, 5 + 1, 1):
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
frame_intense_2 = cv2.warpAffine(frame_intense_2.astype(np.uint8), M, (width, height))
# cv2.imshow("trans_img", frame_intense_2)
# cv2.waitKey(0)
frame_intense_2 = frame_intense_2.astype(np.int32)

# code_img = frame_Lab[:, :, 0] - org_frame_lightness
sub_img = frame_intense_2 - frame_intense_1

# remove the black border
sub_img[0:np.abs(offset_y), :] = 0
sub_img[-np.abs(offset_y):, :] = 0
sub_img[:, 0:np.abs(offset_x)] = 0
sub_img[:, -np.abs(offset_x):] = 0

code_img_lab = (sub_img - np.min(sub_img)) * 255 / (np.max(sub_img) - np.min(sub_img))
code_img_lab = code_img_lab.astype(np.uint8)

cv2.imshow("code_sub", code_img_lab)
cv2.waitKey(0)

# # ========== detect apriltags =============
# cv2.imshow("code_org", frame)
# cv2.waitKey(0)
detector = TagsDetector()
img = np.array([code_img_lab, code_img_lab, code_img_lab], dtype=np.uint8)
img = img.transpose((1, 2, 0))
flag, results = detector.detect(img)
if flag == True:
    for i, result in enumerate(results):
        print(result)
    print(str(len(results)) + ' apriltags are detected in total!')
else:
    print('No apriltag is detected!')

# ======== img processing =============
ret, code_img_BW = cv2.threshold(code_img_lab, 0, 255, cv2.THRESH_OTSU)
