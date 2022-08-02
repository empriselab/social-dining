
import glob

import numpy as np
import pandas as pd
import os
import json
import ast

def get_openpose_from_json(video_id, start_frame, window_size):
    """
    :param video_id: the id of video you want to get openpose
    :param start_frame: the frame when the bite action starts
    :param window_size: the window size of each sample in seconds
    """

    path = '/home/aa2375/social-dining/data/openpose/' + str(video_id)
    frame_list = os.listdir(path)
    frame_list.sort()
    print("len:", len(frame_list))

    b_x, b_y, b_c = [], [], []
    lh_x, lh_y, lh_c = [], [], []
    rh_x, rh_y, rh_c = [], [], []
    f_x, f_y, f_c = [], [], []

    for i in range(start_frame - 30 * window_size, start_frame):
        with open(path + '/' + frame_list[i]) as f:
            pose_frame = json.load(f)
        if not pose_frame['people']:  # if no body detected:
            body_joints = np.empty(25 * 3)
            body_joints[:] = np.nan
            face_joints = np.empty(70 * 3)
            face_joints[:] = np.nan
        else:
            body_joints = pose_frame['people'][0]['pose_keypoints_2d']
            face_joints = pose_frame['people'][0]['face_keypoints_2d']

        a = 0
        while a < (len(body_joints)):
            b_x.append(body_joints[a])
            b_y.append(body_joints[a + 1])
            b_c.append(body_joints[a + 1])
            a += 3
        a = 0
        while a < (len(face_joints)):
            f_x.append(face_joints[a])
            f_y.append(face_joints[a + 1])
            f_c.append(face_joints[a + 1])
            a += 3

    body_op_x = np.array(b_x).reshape(window_size * 30, -1)
    body_op_y = np.array(b_y).reshape(window_size * 30, -1)
    body_op_c = np.array(b_c).reshape(window_size * 30, -1)

    # remove keypoints on legs and feet
    body_op_x = np.delete(body_op_x, [10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24], 1)
    body_op_y = np.delete(body_op_y, [10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24], 1)
    body_op_c = np.delete(body_op_c, [10, 11, 13, 14, 18, 19, 20, 21, 22, 23, 24], 1)


    face_op_x = np.array(f_x).reshape(window_size * 30, -1)
    face_op_y = np.array(f_y).reshape(window_size * 30, -1)
    face_op_c = np.array(f_c).reshape(window_size * 30, -1)
    # sample = np.concatenate((np.concatenate((np.concatenate((np.concatenate((np.concatenate(
    #     (np.concatenate((np.concatenate((body_op_x, body_op_y), axis=1), lhand_op_x), axis=1), lhand_op_y),
    #     axis=1), rhand_op_x), axis=1), rhand_op_y), axis=1), face_op_x), axis=1), face_op_y), axis=1)
    body_sample = np.concatenate([body_op_x, body_op_y], axis=1)
    face_sample = np.concatenate([face_op_x, face_op_y], axis=1)
    return np.array(body_sample).astype(float), np.array(face_sample).astype(float)


def mapping_op_sample_to_elan_label(elan_label_path, out_folder='positive_samples'):
    elan_label = pd.read_csv(elan_label_path, index_col=0)
    op_sample_list = []
    op_sample_1_list = []
    op_sample_2_list = []
    vid_1_list = []
    vid_2_list = []
    for i in range(len(elan_label)):
        vid_main = elan_label['Name'].iloc[i]
        print(vid_main)
        start_frame = elan_label['Start Frame'].iloc[i]
        print(start_frame)

        # this determines the left and right people!
        if vid_main[-1] == '1':
            vid_1_list.append(vid_main[0:-1]+str(2))
            vid_2_list.append(vid_main[0:-1] + str(3))
        elif vid_main[-1] == '2':
            vid_1_list.append(vid_main[0:-1] + str(1))
            vid_2_list.append(vid_main[0:-1] + str(3))
        elif vid_main[-1] == '0':
            continue
        else:
            vid_1_list.append(vid_main[0:-1] + str(1))
            vid_2_list.append(vid_main[0:-1] + str(2))
        op_sample = get_openpose_from_json(vid_main, start_frame, 6)
        op_name = f"data/{out_folder}/" + str(i).zfill(5) + '_main.npy'
        np.save(op_name, op_sample)
        op_sample_list.append(op_name)
        op_1_sample = get_openpose_from_json(vid_1_list[i], start_frame, 6)
        op_1_name = f"data/{out_folder}/" + str(i).zfill(5) + '_1.npy'
        np.save(op_1_name, op_1_sample)
        op_sample_1_list.append(op_1_name)
        op_2_sample = get_openpose_from_json(vid_2_list[i], start_frame, 6)
        op_2_name = f"data/{out_folder}/" + str(i).zfill(5) + '_2.npy'
        np.save(op_2_name, op_2_sample)
        op_sample_2_list.append(op_2_name)
    elan_label['video_id_1'] = vid_1_list
    elan_label['video_id_2'] = vid_2_list
    elan_label['op_main'] = op_sample_list
    elan_label['op_1'] = op_sample_1_list
    elan_label['op_2'] = op_sample_2_list
    elan_label.to_csv(f"data/{out_folder}_with_op.csv")


def interpolate_gaze_headpose():
    df = pd.read_csv("/Users/tongwu/Downloads/rt_gene_feats.csv", usecols=['name', 'gaze'])
    l = []
    for i in range(1, len(df) - 1):
        pre = df.iloc[i-1, 0][-5:]
        cur = df.iloc[i, 0][-5:]
        if int(pre) != int(cur) - 1 and df.iloc[i-1, 0][:3] ==df.iloc[i, 0][:3]:
            diff = int(cur) - int(pre)
            for j in range(diff):
                should = int(pre) + j
                l.append(df.iloc[i-1, 0][:-5] + str(should).zfill(5))
        else:
            l.append(df.iloc[i-1, 0])
    new_col = pd.DataFrame({"new_name": l})
    df = df.set_index("name")
    new_col = new_col.set_index("new_name")
    df_new = new_col.join(df)
    df_new = df_new.interpolate()
    print()

def get_gaze_from_csv(video_id, start_frame, window_size):
    """
        :param video_id: the id of video you want to get openpose
        :param start_frame: the frame when the bite action starts
        :param window_size: the window size of each sample in seconds
        """
    gaze_df = pd.read_csv(f"data/rt_gene/{video_id}.csv")
    gaze_df = gaze_df.set_index('name')
    start_id = video_id + '_' + str(start_frame).zfill(5)
    end_id = video_id + '_' + str(start_frame + 30 * window_size).zfill(5)
    data = gaze_df.loc[start_id : end_id]
    gaze = np.array(data['gaze'])
    head_pose = np.array(data['headpose'])
    for i in range(len(gaze)):
        gaze[i] = np.array(ast.literal_eval(gaze[i]))
        head_pose[i] = np.array(ast.literal_eval(head_pose[i]))
    return np.array(gaze), np.array(head_pose)


# load positive csv

# add openposes to it
mapping_op_sample_to_elan_label('data/positive_labels.csv', 'positive_samples')

# add gazes to it

# add speaking to it

# add count to it


# load negative csv

# add openposes to it
mapping_op_sample_to_elan_label('data/negative_labels.csv', 'negative_samples')
