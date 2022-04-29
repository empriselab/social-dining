import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

path = "/Users/tongwu/Downloads/03_1.mp4/"

frame_list = os.listdir(path)
frame_list.sort()
values = []
for i in range (len(frame_list)):
    b_x, b_y, b_c = [], [], []
    lh_x, lh_y, lh_c = [], [], []
    rh_x, rh_y, rh_c = [], [], []
    f_x, f_y, f_c = [], [], []

    with open(path + frame_list[i]) as f:
        pose_frame = json.load(f)
    if not pose_frame['people']:  # if no body detected:
        body_joints = np.empty(25 * 3)
        body_joints[:] = np.nan
        lhand_joints = np.empty(21 * 3)
        lhand_joints[:] = np.nan
        rhand_joints = np.empty(21 * 3)
        rhand_joints[:] = np.nan
        face_joints = np.empty(70 * 3)
        face_joints[:] = np.nan
    else:
        body_joints = pose_frame['people'][0]['pose_keypoints_2d']
        lhand_joints = pose_frame['people'][0]['hand_left_keypoints_2d']
        rhand_joints = pose_frame['people'][0]['hand_right_keypoints_2d']
        face_joints = pose_frame['people'][0]['face_keypoints_2d']

    a = 0
    while a < (len(face_joints)):
        f_x.append(face_joints[a])
        f_y.append(face_joints[a + 1])
        f_c.append(face_joints[a + 1])
        a += 3

    up = f_y[66]
    down = f_y[62]
    value = up - down
    values.append(value)
values = np.array(values)
values = np.nan_to_num(values)
for i in range(len(values)):
    if values[i] < -5:
        values[i] = 0
num = 0
for i in range(len(values)):
    if values[i] > 25 and values[i] < 35:
        print(i)
        num += 1
plt.plot(values)
plt.show()
print(num)
