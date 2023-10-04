import numpy as np
from tqdm import tqdm
import argparse

from helper_code import load_participant_data, find_participant_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='DEAP/data', help='path to the data folder that contains data for all participants')
    parser.add_argument('--task', type=str, default='bi_class', help='task type, bi_class or regression')
    args = parser.parse_args()
    data_folder = args.data_folder
    task = args.task

    data_folder = data_folder[:-1] if data_folder.endswith('/') else data_folder
    participant_ids = find_participant_ids(data_folder)
    print('participant_ids', participant_ids)
    video_features = []
    eeg_features = []
    other_physio_features = list()
    labels = list()
    print('loading data...')
    for participant_id in tqdm(participant_ids):
        for trial_id in range(1, 41):
            trial_id = '{:02d}'.format(trial_id)
            data_dict = load_participant_data(data_folder, participant_id, trial_id, task=task)
            label = data_dict['labels'] # ['valence', 'arousal', 'dominance', 'liking']
            video = data_dict['video']
            eeg_data = data_dict['eeg_data']
            other_physio_data = data_dict['other_physio_data']
            if video is not None:
                labels.append(label)
                video_features.append(video[np.newaxis, :])
                eeg_features.append(eeg_data[np.newaxis, :])
                other_physio_features.append(other_physio_data[np.newaxis, :])
            else:
                print('\tparticipant {0}, trial {1} has no video data'.format(participant_id, trial_id))
            
    labels = np.vstack(labels)
    video_features = np.vstack(video_features)
    eeg_features = np.vstack(eeg_features)
    other_physio_features = np.vstack(other_physio_features)
    print('\tlabels:', labels.shape)
    print('\tvideo_features:', video_features.shape)
    print('\teeg_features:', eeg_features.shape)
    print('\tother_physio_features:', other_physio_features.shape)