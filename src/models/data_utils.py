import numpy as np
import tqdm



def organize_data(person_features, global_features, use_ssp=0, half_frame_length=False):
    samples = np.load('/home/aa2375/social-dining/src/preprocessing/all_samples.npy', allow_pickle=True)
    person_main = []
    person_left = []
    person_right = []
    global_feats = []
    video_ids = []
    issue_count = 0
    
    labels = []
    
    for i in tqdm.tqdm(range(len(samples))):
        
        if samples[i]['start_frame'] < 0: # this should happen 16 times
            issue_count += 1
            continue
     
        if ('speaking' in person_features) and ('09' in samples[i]['main_id']):
            continue

        person_main.append([])
        person_left.append([])
        person_right.append([])
        global_feats.append([])
        labels.append(samples[i]['label'])
        video_ids.append(samples[i]['main_id'])

        for feature in person_features:          
            main_feature = samples[i]['main'][feature]
            left_feature = samples[i]['left'][feature]
            right_feature = samples[i]['right'][feature]
            
            if len(left_feature) != 180:
                # this only happens once in 14_2, so we ensure the rest of the values are the same...
                # just grow the size of it to 180
                print(feature, samples[i]['main_id'], left_feature.shape)
                diff = 180 - len(left_feature)
                to_add = left_feature[-1].reshape(1, -1).repeat(diff, axis=0)
                left_feature = np.concatenate([left_feature, to_add])
            if use_ssp:
                # Social Signal Processing-like task
                # Use the first frame for all the frames if feature is body
                if feature == 'body':
                    main_feature[:] = main_feature[0]

            # # if half_frame_length is true, we will skip every other frame!
            if half_frame_length:
                main_feature = main_feature[::2]
                left_feature = left_feature[::2]
                right_feature = right_feature[::2]

            # currently breaking things as a test. uncomment later            
            person_main[-1].append(main_feature)
            person_left[-1].append(left_feature)
            person_right[-1].append(right_feature)


        person_main[-1] = np.concatenate(person_main[-1], axis=1)
        person_left[-1] = np.concatenate(person_left[-1], axis=1)
        person_right[-1] = np.concatenate(person_right[-1], axis=1)

        for feature in global_features:
            global_feats[-1].append(samples[i]['main'][feature])
    print(issue_count)
    return np.array(person_main), np.array(person_left), np.array(person_right), np.array(global_feats), np.array(labels), np.array(video_ids)

audio_feature_names = ['speaking']
video_feature_names = ['gaze', 'headpose', 'body', 'face']
time_feature_names = ['time_since_last_bite', 'time_since_start']
count_feature_names = ['num_bites']


def get_feature_dicts(person_features, global_features, use_ssp=False, half_frame_length=False):
    samples = np.load('/home/aa2375/social-dining/src/preprocessing/all_samples.npy', allow_pickle=True)

    audio_inputs = [[] for i in range(3)]
    video_inputs = [[] for i in range(3)]
    time_inputs = []
    count_inputs = []
    video_ids = []

    labels = []
    
    for i in tqdm.tqdm(range(len(samples))):
        
        if samples[i]['start_frame'] < 0: # this should happen 16 times
            continue
     
        if ('speaking' in person_features) and ('09' in samples[i]['main_id']):
            continue


        any_audio = False
        any_video = False
        any_count = False
        any_time = False

        for feature in person_features:
            if feature in audio_feature_names:
                any_audio = True
            if feature in video_feature_names:
                any_video = True
        for feature in global_features:
            if feature in count_feature_names:
                any_count = True
            if feature in time_feature_names:
                any_time = True
        


        for k in range(3):
            audio_inputs[k].append([])
            video_inputs[k].append([])

        time_inputs.append([])
        count_inputs.append([])

        labels.append(samples[i]['label'])
        video_ids.append(samples[i]['main_id'])

        for feature in person_features:
            main_feature = samples[i]['main'][feature]
            left_feature = samples[i]['left'][feature]
            right_feature = samples[i]['right'][feature]
            
            if len(left_feature) != 180:
                # this only happens once in 14_2, so we ensure the rest of the values are the same...
                # just grow the size of it to 180
                print(feature, samples[i]['main_id'], left_feature.shape)
                diff = 180 - len(left_feature)
                to_add = left_feature[-1].reshape(1, -1).repeat(diff, axis=0)
                left_feature = np.concatenate([left_feature, to_add])

            if use_ssp:
                # Social Signal Processing-like task
                # Use the first frame for all the frames if feature is body
                if feature == 'body':
                    main_feature[:] = main_feature[0]

            # if half_frame_length is true, we will skip every other frame!
            if half_frame_length:
                main_feature = main_feature[::2]
                left_feature = left_feature[::2]
                right_feature = right_feature[::2]

            if feature == 'body' or feature == 'face' or feature == 'gaze' or feature == 'headpose':
                video_inputs[0][-1].append(main_feature)
                video_inputs[1][-1].append(left_feature)
                video_inputs[2][-1].append(right_feature)

            if 'audio' in feature or feature == 'speaking':
                audio_inputs[0][-1].append(main_feature)
                audio_inputs[1][-1].append(left_feature)
                audio_inputs[2][-1].append(right_feature)

        frame_count = 180 if not half_frame_length else 90
        # frame_count = 180

        for k in range(3):
            if any_audio:
                audio_inputs[k][-1] = np.concatenate(audio_inputs[k][-1], axis=1)
            else:
                audio_inputs[k][-1] = np.zeros((frame_count, 0))
            if any_video:
                video_inputs[k][-1] = np.concatenate(video_inputs[k][-1], axis=1)
            else:
                video_inputs[k][-1] = np.zeros((frame_count, 0))




        for feature in global_features:
            if any_time and feature in time_feature_names:
                time_inputs[-1].append(samples[i]['main'][feature])
            if any_count and feature in count_feature_names:
                count_inputs[-1].append(samples[i]['main'][feature])
            

    return np.array(audio_inputs), np.array(video_inputs), np.array(time_inputs), np.array(count_inputs), np.array(labels), np.array(video_ids)

# feats = ['body', 'face', 'gaze', 'headpose', 'speaking']
# global_feats = ['num_bites', 'time_since_last_bite']
# a, b, t, c, l, v = get_feature_dicts(feats, global_feats)

# print(a.shape)
# print( b.shape)
# print( t.shape)
# print( c.shape)
# print( l.shape)
# print( v.shape)


# import time
# start = time.time()
# # main, left, right, global_feats = organize_data(['body', 'face', 'gaze', 'headpose', 'speaking'], ['num_bites', 'time_since_last_bite'])
# main, left, right, global_feats = organize_data(['face', 'gaze', 'headpose', 'speaking'], ['num_bites', 'time_since_last_bite'])