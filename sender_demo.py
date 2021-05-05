#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: sender_demo.py
Date: 2021/5/4 20:48
LastEditors: LI Jinjie
LastEditTime: 2021/5/4 20:48
Description: a small demo to test the threshold of delta light intensity of L
'''

import cv2
import numpy as np
import time
import copy


def main():
    # ===== make a new video =======
    video_name = 'test_videos/output.avi'
    cap = cv2.VideoCapture(video_name)
    org_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    org_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_videos/output_masked.avi', fourcc, 60.0, (int(org_width), int(org_height)))

    # ======== apriltags map related ============
    apriltag_map = cv2.imread('apriltag_map/apriltagMap_7x7_6.7000m.png', flags=cv2.IMREAD_GRAYSCALE)
    [width, height] = apriltag_map.shape
    map_mask = np.zeros([height, width], dtype=np.uint8)
    map_mask[apriltag_map < 100] = 1

    mask = np.zeros([int(org_height), int(org_width)], dtype=np.uint8)

    for i in range(min(mask.shape[0], map_mask.shape[0])):
        for j in range(min(mask.shape[1], map_mask.shape[1])):
            mask[i][j] = map_mask[i][j]

    # cv2.imshow("a", mask * 255)
    # cv2.waitKey(0)

    DELTA_L = 8  # an essential value
    sign = 1
    tmp = 0
    time_start = time.time()

    print("Processing......")
    while cap.isOpened():
        tmp += 1
        sign = -sign
        ret, frame = cap.read()
        if ret is True:
            frame_Lab = cv2.cvtColor(frame, code=cv2.COLOR_BGR2Lab)  # transform from BGR to CIELAB

            lightness_masked = frame_Lab[:, :, 0].astype(np.int32)
            lightness_masked += sign * DELTA_L * mask
            lightness_masked[lightness_masked > 255] = 255  # pay attention to the value that more than 255.
            frame_Lab[:, :, 0] = lightness_masked.astype(np.uint8)

            frame_masked = cv2.cvtColor(frame_Lab, code=cv2.COLOR_Lab2BGR)  # transform from CIELAB to BGR

            # cv2.imshow("a", frame_masked)
            # cv2.waitKey(0)

            out.write(frame_masked)

        else:
            break

        if tmp == 1500:  # 60 fps, 25 s
            time_end = time.time()
            break
        # Release everything if job is finished
    cap.release()
    out.release()

    print("Finished!")
    print("Processing time for one round is", (time_end - time_start) / 1500)


def freq_30_to_60():
    video_name = 'test_videos/map_record.mp4'
    cap = cv2.VideoCapture(video_name)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_videos/output.avi', fourcc, 60.0, (int(width), int(height)))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            out.write(frame)
            out.write(frame)  # 倍频
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()


if __name__ == "__main__":
    # freq_30_to_60()
    main()
