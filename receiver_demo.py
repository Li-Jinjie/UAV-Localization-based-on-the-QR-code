#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: LI Jinjie
File: receiver_demo.py
Date: 2021/5/5 19:13
LastEditors: LI Jinjie
LastEditTime: 2021/5/5 19:13
Description: a demo of receiver to post-process the videos
'''

import cv2
import numpy as np
import time
import copy
from apriltags_detector_new import TagsDetector


def main():
    detector = TagsDetector()
    # ===== open a video =======
    path = 'receiver_videos/'
    video_name = '0521_color_120fps_80d_720p.avi'
    # video_name = '0517_color_120fps_80degree_720p.avi'
    cap = cv2.VideoCapture(path + video_name)
    org_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    org_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('sender_videos/output_masked.avi', fourcc, 60.0, (int(org_width), int(org_height)))

    time_start = time.time()
    cnt = 0

    org_frame_lightness = np.zeros([int(org_height), int(org_width)], dtype=np.uint8)
    last_frame_bgr = np.zeros([int(org_height), int(org_width), 3], dtype=np.uint8)

    last_code_img = np.zeros([int(org_height), int(org_width)], dtype=np.uint8)
    print("Processing......")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:

            # ====== LAB =====

            frame_Lab = cv2.cvtColor(frame, code=cv2.COLOR_BGR2Lab)  # transform from BGR to LAB
            frame_now_d = frame_Lab[:, :, 0].astype(np.int32)
            frame_last_d = org_frame_lightness.astype(np.int32)
            org_frame_lightness = frame_Lab[:, :, 0]

            # frame_gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)  # transform from BGR to gray
            # frame_now_d = frame_gray.astype(np.int32)
            # frame_last_d = org_frame_lightness.astype(np.int32)
            # org_frame_lightness = frame_gray

            # code_img = frame_Lab[:, :, 0] - org_frame_lightness
            sub_img = frame_now_d - frame_last_d
            code_img_lab = (sub_img - np.min(sub_img)) * 255 / (np.max(sub_img) - np.min(sub_img))
            code_img_lab = code_img_lab.astype(np.uint8)

            # cv2.imshow("code_sub", code_img_lab)
            # cv2.waitKey(0)


            # # ========== detect apriltags =============
            if cnt > 250:
                # cv2.imshow("code_org", frame)
                # cv2.waitKey(0)

                flag, results = detector.detect(code_img_lab)
                if flag == True:
                    for i, result in enumerate(results):
                        print(result)
                    print(str(len(results)) + ' apriltags are detected in total!')
                else:
                    print('No apriltag is detected!')

                # ======== img processing =============
                ret, code_img_BW = cv2.threshold(code_img_lab, 0, 255, cv2.THRESH_OTSU)

            print(cnt)

            code_img_add = (code_img_lab.astype(np.int32) + last_code_img) / 2
            code_img_add = (code_img_add - np.min(code_img_add)) * 255 / (np.max(code_img_add) - np.min(code_img_add))
            last_code_img = code_img_lab.astype(np.int32)

            # cv2.imshow("code_add", code_img_add.astype(np.uint8))
            # cv2.waitKey(0)

            # ===== BGR ====
            # frame = frame.astype(np.double)
            # last_frame_bgr = last_frame_bgr.astype(np.double)
            #
            # sub_img = frame - last_frame_bgr
            #
            # code_img_bgr = (sub_img - np.min(sub_img)) / (np.max(sub_img) - np.min(sub_img))
            #
            # code_img_bgr = code_img_bgr.astype(np.uint8)

            if cnt == -1:
                # cv2.imshow("code_add", code_img_add.astype(np.uint8))
                # cv2.waitKey(0)
                name = video_name + "_lab.png"
                # cv2.imwrite("receiver_pictures/" + name, code_img_lab.astype(np.uint8))

                cv2.imshow("code_lab", code_img_lab)
                cv2.imshow("code_lab", code_img_lab.astype(np.uint8))
                cv2.waitKey(0)
                cv2.imwrite("receiver_pictures/" + name, code_img_lab.astype(np.uint8))
                cv2.imshow("code_BW", code_img_BW)
                cv2.imshow("frame", frame)
                cv2.waitKey(0)
            #     cv2.imshow("code", code_img_bgr)
            #     cv2.imshow("img_last", last_frame_bgr)
            #     cv2.imshow("img_now", frame)
            #     cv2.waitKey(0)

            # cv2.imwrite("frame_last", last_frame_bgr)
            # cv2.imwrite("frame_now", frame)

            cnt += 1

            last_frame_bgr = frame

        else:
            break

        # if tmp == 1500:  # 60 fps, 25 s
        #     time_end = time.time()
        #     break
        # Release everything if job is finished
    cap.release()

    print("Finished!")


if __name__ == "__main__":
    main()
