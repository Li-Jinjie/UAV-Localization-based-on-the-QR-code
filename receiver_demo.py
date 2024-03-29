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
import csv
from projected_tag.detector import ProjectedTagsDetector


def main():
    path_map = "tag_map\\maps_info.yaml"
    path_camera_para = "data_real\\20210530_full_data\\camera_calibration_data\\GZ120_grid_size=20mm.npz"
    detector = ProjectedTagsDetector(path_map, path_camera_para, detector_type='AprilTag3')
    # detector_type: 'AprilTag3', 'ByMe', 'ArUcoOpenCV'
    # ========= open a video ============
    path = 'data_real/20210530_full_data/'
    video_name = '0623_color_120fps_L=4_9x9_noLight_720p_UWB_opt_1.avi'
    # path = 'data_real/20210623_full_data/'    'data_real/20210530_full_data/'
    # video_name = '0623_color_120fps_L=4_9x9_noLight_720p_UWB_opt_2.avi'
    # video_name = '0530_color_120fps_L=4_9x9_noLight_720p_with_optitrack.avi'
    cap = cv2.VideoCapture(path + video_name)
    org_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    org_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('sender_videos/output_masked.avi', fourcc, 60.0, (int(org_width), int(org_height)))

    # ========= open a csv file =========
    with open(path + '1_data_tag_0623_ofc_no_cor_all_pts.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, )
        time_start = time.time()
        cnt = 0
        print("Processing......")
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                # # ========== detect apriltags =============
                if cnt > -1:  # 1262 for the specific frame
                    flag, results = detector.detect(frame)
                    rvec, tvec = detector.estimate_pose(flag, results, estimate_method='all_pts', ransac_flag=False)
                    # all_pts, average
                    if tvec is not None:
                        # ========= write to a csv file =========
                        csv_writer.writerow([cnt, cnt / fps, tvec.item(0), tvec.item(1), tvec.item(2),
                                             rvec.item(0), rvec.item(1), rvec.item(2)])
                        pass

                        print("xyz=", tvec)
                        # for i, result in enumerate(results):
                        #     print(result)
                        # print(str(len(results)) + ' apriltags are detected in total!')
                    else:
                        pass
                        # print('No apriltag is detected!')

                print(cnt)
                cnt += 1
            else:
                break
            time_end = time.time()
        cap.release()
        print("Finished!")
        print("Processing time: ", time_end - time_start, " s")


if __name__ == "__main__":
    main()
