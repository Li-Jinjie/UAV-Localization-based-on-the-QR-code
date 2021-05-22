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
    video_name = 'sender_videos/output.avi'
    cap = cv2.VideoCapture(video_name)
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # ======== apriltags map related ============
    apriltag_map = cv2.imread('apriltag_map/apriltagMap_3x3_1920.0000m.png', flags=cv2.IMREAD_GRAYSCALE)
    [height, width] = apriltag_map.shape
    map_mask = np.zeros([height, width], dtype=np.int32)
    # white: 255 > 1 black: 0 > -1 no color: 127 > 0
    map_mask[apriltag_map >= 255] = 1  # white
    map_mask[apriltag_map <= 0] = -1  # black

    # mask = np.zeros([int(video_height), int(video_width)], dtype=np.uint8)
    #
    # for i in range(min(mask.shape[0], map_mask.shape[0])):
    #     for j in range(min(mask.shape[1], map_mask.shape[1])):
    #         mask[i][j] = map_mask[i][j]

    # cv2.imshow("a", mask * 255)
    # cv2.waitKey(0)

    # ======= record videos ============
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('sender_videos/square_masked.avi', fourcc, 60.0, (int(width), int(height)))

    DELTA_L = 5  # an essential value
    sign = 1
    tmp = 0
    time_start = time.time()

    print("Processing......")
    while cap.isOpened():
        tmp += 1
        sign = -sign
        ret, frame = cap.read()
        if ret is True:
            # change 1920*1440 to 1920*2160
            frame_small = frame[int(video_height / 2 - 960 / 2):int(video_height / 2 + 960 / 2),
                        int(video_width / 2 - 1080 / 2):int(video_width / 2 + 1080 / 2)]

            frame_large = cv2.resize(frame_small, (width, height))

            frame_Lab = cv2.cvtColor(frame_large, code=cv2.COLOR_BGR2Lab)  # transform from BGR to CIELAB

            lightness_masked = frame_Lab[:, :, 0].astype(np.int32)
            lightness_masked += sign * DELTA_L * map_mask
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
    video_name = 'sender_videos/map_record.mp4'
    cap = cv2.VideoCapture(video_name)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('sender_videos/output.avi', fourcc, 60.0, (int(width), int(height)))

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
