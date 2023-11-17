import json
import pathlib
import csv
import numpy as np
from scipy.interpolate import interp1d
import math
import cv2

# NOTE: change the directories here!


# These directories are the inputs provided by our dataset
gaze_dir = '/home/tangyimi/social_signal/dining_dataset/processed_gazes/'
status_dir = '/home/tangyimi/social_signal/dining_dataset/upsampled-person-speaking/'
keypoints_dir = '/home/tangyimi/social_signal/vision_openpose_features/'
# word_dir = 'dining_dataset/words/v1/' # not needed

# These are easier-to-use preprocessed output directories
process_gazepose_dir = '/home/tangyimi/social_signal/dining_dataset/full_gazes/'
process_keypoints_dir = '/home/tangyimi/social_signal/dining_dataset/full_keypoints/'
clean_keypoints_dir = '/home/tangyimi/social_signal/dining_dataset/clean_keypoints/'

def calculate_length():
    result = []

    for i in range(30):
        if i+1 == 9:
            result.append(0)
            continue

        status_path = status_dir + '{:02d}'.format(i+1) + '.npy'
        status_array = np.load(status_path)

        # word_path = word_dir + '{:02d}'.format(i+1) + '.jsonl'
        

        # with open(word_path, 'r') as f:
        #     for word_length, _ in enumerate(f, start=1):
        #         pass

        # assert len(status_array) == word_length, 'File: {} status length: {}, word length: {}'.format(status_path, len(status_array), word_length)

        result.append(len(status_array))

    return result


def check_length(length_list, is_keypoint=True):
    print('Checking...')
    count = 0
    for i in range(30):
        if i+1 == 9:
            continue
        frame_length = length_list[i]
        
        if is_keypoint:
            for person in range(3):
                person_dir = pathlib.Path(keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '/')
                for keypoint_length, file in enumerate(sorted(person_dir.iterdir()), start=1):
                    pass
            
                if keypoint_length != frame_length:
                    count += 1
                    print('File ({}) inconsistent, word_length: {}, keypoint_length: {}'.format(person_dir, frame_length, keypoint_length))
        
        else:
            for person in range(3):
                gaze_path = gaze_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.csv'
                with open(gaze_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for gaze_length, row in enumerate(reader, start=1):
                        pass
                        
                if frame_length != gaze_length:
                    count += 1
                    print('File ({}) inconsistent, word_length: {}, gaze_length: {}'.format(gaze_path, frame_length, gaze_length))
        
    print('total inconsistent file:', count)


def process_gazepose(length_list):
    print('Processing gazepose...')
    for i in range(30):
        if i+1 == 9:
            continue

        frame_length = length_list[i]

        for person in range(3):
            headpose_list = np.zeros((frame_length, 2))
            gaze_list = np.zeros((frame_length, 2))
            gaze_path = gaze_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.csv'
            print(gaze_path)

            with open(gaze_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    index = int(row['name'])
                    headpose = row['headpose']
                    gaze = row['gaze']

                    if index < frame_length:
                        headpose_list[index] = np.array(eval(headpose))
                        gaze_list[index] = np.array(eval(gaze))
            #count = 0
            for index in range(frame_length):
                if headpose_list[index][0] == 0 and headpose_list[index][1] == 0:
                    #count += 1
                    #print(index)
                    if index == 0:
                        replace = index + 1
                        while headpose_list[replace][0] == 0 and headpose_list[replace][1] == 0:
                            replace += 1

                        headpose_list[index] = headpose_list[replace]
                        gaze_list[index] = gaze_list[replace]
                    else:
                        replace = index - 1

                        headpose_list[index] = headpose_list[replace]
                        gaze_list[index] = gaze_list[replace]

            assert not any((headpose_list == np.zeros(2)).all(1)) and \
                not any((gaze_list == np.zeros(2)).all(1)), 'File: {} inconsistent, empty file not handled'.format(gaze_path)
            #print('total empty:', count)
            #print('start writing')

            process_path = process_gazepose_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'

            np.savez(process_path, headpose=np.array(headpose_list), gaze=np.array(gaze_list))

                    

def process_keypoints(length_list):
    print('Processing keypoints...')
    for i in range(30):
        if i+1 == 9:
            continue

        frame_length = length_list[i]
        

        for person in range(3):
            pose_list = np.zeros((frame_length, 75))
            #face_list = np.zeros((frame_length, 210))
            empty = []

            person_dir = pathlib.Path(keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '/')
            print(person_dir)

            for keypoint_length, file in enumerate(sorted(person_dir.iterdir()), start=0):
                assert keypoint_length == int(file.name.split('_')[2]), 'Missing file: {} {}'.format(keypoint_length, person+1)
                
                with open(file, 'r') as f:
                    keypoints = json.load(f)['people']


                if keypoints != []:
                    pose = keypoints[0]['pose_keypoints_2d']
                    face = keypoints[0]['face_keypoints_2d']
                    
                    if pose == []:
                        print('pose empty')
                        print(file)
                    else:
                        if keypoint_length < frame_length:
                            pose_list[keypoint_length] = np.array(pose)

                    if face != []:
                        print('face not empty')
                        print(file)
            
            # handle empty files
            for index in range(frame_length):
                if sum(pose_list[index]) == 0:
                    if index == 0:
                        replace = index + 1
                        while sum(pose_list[replace]) == 0:
                            replace += 1

                        pose_list[index] = pose_list[replace]
                        #face_list[index] = face_list[replace]
                    else:
                        replace = index - 1

                        pose_list[index] = pose_list[replace]
                        #face_list[index] = face_list[replace]
            
            assert not any((pose_list == np.zeros(75)).all(1)), 'File: {} inconsistent, empty file not handled'.format(person_dir)
            #print('start writing')

            # write to jsonline file for each person and each video
            process_path = process_keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'

            np.savez(process_path, pose=pose_list)
            


def index_pose(array):
     # 9, 10, 11, 12
    indices = [0,1,2,3,4,5,6,7,8,15,16,17,18]
    indices_xy = [j for i in indices for j in (i*3, i*3 + 1)] 

    return array[:, indices_xy]

def clean_pose():
    # read npz file
    for i in range(30):
        if i+1 == 9:
            continue

        for person in range(3):
            file_name = process_keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'
            print('Cleaning file: {}'.format(file_name))
            pose = np.load(file_name)['pose']
            new_pose = index_pose(pose)
            

            for column in range(0, new_pose.shape[-1]):
                valid_entries = np.nonzero(new_pose[:, column])[0]
                missing_entries = np.where(new_pose[:, column] == 0)[0]

                interp_func = interp1d(valid_entries, new_pose[valid_entries, column], bounds_error=False)

                new_pose[missing_entries, column] = interp_func(missing_entries)
                
                first_non_zero = new_pose[valid_entries[0], column]
                last_non_zero = new_pose[valid_entries[-1], column]
                new_pose[:valid_entries[0]] = first_non_zero
                new_pose[valid_entries[-1] + 1:] = last_non_zero
                
                assert not any((new_pose[:, column] == 0)), 'File: {} column: {} has 0'.format(file_name, column)
                
            # write to jsonline file for each person and each video
            clean_file = clean_keypoints_dir + '{:02d}_{:d}'.format(i+1, person+1) + '.npz'
            np.savez(clean_file, pose=new_pose)

            


#### Visualization functions ####

def create_image(width, height):
    blank_image = np.zeros((height, width, 3), np.uint8)
    blank_image[:] = (255, 255, 255)
    return blank_image

def get_endpoint(theta, phi, center_x, center_y, length=300):
    endpoint_x = -1.0 * length * math.cos(theta) * math.sin(phi) + center_x
    endpoint_y = -1.0 * length * math.sin(theta) + center_y
    return endpoint_x, endpoint_y


def visualize_headgaze(image, est_gaze,color=(255,0,0)):
    output_image = np.copy(image)
    center_x = output_image.shape[1] / 2
    center_y = output_image.shape[0] / 2

    endpoint_x, endpoint_y = get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 100)

    cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), color, 2)
    bordered_image = cv2.copyMakeBorder(output_image, top=5, bottom=5, left=5, right=5, 
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return bordered_image


def get_xy(pose, index):
    return (int(pose[index*2]), int(pose[index*2+1]))


def visualize_pose(image, pose):  
    thickness = 4 

    cv2.line(image, get_xy(pose, 0), get_xy(pose, 1), (51,0,153), thickness)
    cv2.line(image, get_xy(pose, 0), get_xy(pose, 9), (102,0,153), thickness)
    cv2.line(image, get_xy(pose, 0), get_xy(pose, 10), (153,0,102), thickness)
    cv2.line(image, get_xy(pose, 1), get_xy(pose, 2), (1,51,153), thickness)
    cv2.line(image, get_xy(pose, 1), get_xy(pose, 5), (0,153,102), thickness)
    cv2.line(image, get_xy(pose, 1), get_xy(pose, 8), (1,0,153), thickness)
    cv2.line(image, get_xy(pose, 2), get_xy(pose, 3), (1,102,154), thickness)
    cv2.line(image, get_xy(pose, 3), get_xy(pose, 4), (0,153,153), thickness)
    cv2.line(image, get_xy(pose, 5), get_xy(pose, 6), (0,153,51), thickness)
    cv2.line(image, get_xy(pose, 6), get_xy(pose, 7), (0,153,0), thickness)
    cv2.line(image, get_xy(pose, 9), get_xy(pose, 11), (153,0,153), thickness)
    cv2.line(image, get_xy(pose, 10), get_xy(pose, 12), (153,0,51), thickness)
    bordered_image = cv2.copyMakeBorder(image, top=5, bottom=5, left=5, right=5, 
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return bordered_image



def write_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()



def construct_batch_video(batch, task, color=(255,0,0)):
    # batch shape (16, 3, 180, 2)
    batch_frames = []
    for i in range(batch.shape[0]):
        each_batch = batch[i]
        frames = []

        for j in range(each_batch.shape[1]):
            people_image = []

            for k in range(each_batch.shape[0]):
                person = each_batch[k][j]
                
                if task == 'pose':
                    blank = create_image(600, 500)
                    image = visualize_pose(blank, person)
                else:
                    blank = create_image(250, 250)
                    image = visualize_headgaze(blank, person, color)

                people_image.append(image)

            frames.append(np.concatenate(people_image, axis=1))
        batch_frames.append(frames)

    return np.array(batch_frames)


def save(y, type, file_name):
    output = 0
    fps = 30 # could be 15 if we downsampled

    result_videos = construct_batch_video(y, type, color=(0,0,255))

    # result_videos = np.concatenate([videos_prediction, videos_inference], axis=2)

    for i in range(len(result_videos)):
        new_file_name = file_name + '_' + str(output) + '.mp4'
        write_video(result_videos[i], new_file_name, fps)
        output += 1


if __name__ == '__main__':

    # NOTE: Run preprocessing first
    # print('Preprocessing...')
    # total_length = calculate_length()
    # check_length(total_length, True)
    # process_gazepose(total_length)
    # process_keypoints(total_length)
    # clean_pose()


    print("Saving pose...")

    # let us load one keypoint file
    p1 = np.load(clean_keypoints_dir + '01_1.npz')
    p1 = p1['pose'] # this is a numpy array of shape (N, 26)

    p2 = np.load(clean_keypoints_dir + '01_2.npz')
    p2 = p2['pose'] # this is a numpy array of shape (N, 26)

    p3 = np.load(clean_keypoints_dir + '01_3.npz')
    p3 = p3['pose'] # this is a numpy array of shape (N, 26)

    # stack so that it is 1 x 3 x N x 26
    pose = np.stack((p1, p2, p3), axis=0)
    pose = np.expand_dims(pose, axis=0)

    # let us only render the first 300 frames (10 seconds)
    pose = pose[:, :, :300, :]

    
    save(pose, 'pose', 'temp_pose')
    print('Pose saved')

    print("Saving gaze...")


    # let us load one gaze file
    g1 = np.load(process_gazepose_dir + '01_1.npz')
    g1 = g1['gaze'] # this is a numpy array of shape (N, 2)

    g2 = np.load(process_gazepose_dir + '01_2.npz')
    g2 = g2['gaze'] # this is a numpy array of shape (N, 2)

    g3 = np.load(process_gazepose_dir + '01_3.npz')
    g3 = g3['gaze'] # this is a numpy array of shape (N, 2)

    # stack so that it is 1 x 3 x N x 2
    gaze = np.stack((g1, g2, g3), axis=0)
    gaze = np.expand_dims(gaze, axis=0)

    # let us only render the first 300 frames (10 seconds)
    gaze = gaze[:, :, :300, :]

    save(gaze, 'gaze', 'temp_gaze')
    print('Gaze saved')

    

