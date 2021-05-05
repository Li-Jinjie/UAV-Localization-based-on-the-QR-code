#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: signal_amp_experiment.py
Date: 2021/5/4 20:48
LastEditors: LI Jinjie
LastEditTime: 2021/5/4 20:48
Description: a small demo to test the threshold of delta light intensity of L
'''

import cv2
import numpy as np
import time


def main():
    img = np.ones([1080, 1920], dtype=np.uint8) * 120
    delta_L = 1
    sign = 1
    tmp = 0
    time_start = time.time()
    while True:
        tmp += 1
        sign = -sign
        img[:, 960:] = img[:, 960:] + sign * delta_L
        cv2.imshow("Image", img)
        cv2.waitKey(int(5))
        # if tmp == 100:
        #     time_end = time.time()
        #     break
    print("Processing time for one round is", (time_end - time_start)/100)

def freq_30_to_60():
    video_name = 'test_videos/map_record.mp4'
    cap = cv2.VideoCapture(video_name)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fps = cap.get(cv2.CAP_PROP_FPS) * 2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_videos/output.avi', fourcc, 60.0, (int(width), int(height)))

    while(cap.isOpened()):
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

