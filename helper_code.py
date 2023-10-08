import scipy as sp
import numpy as np
import cv2, os
from tqdm import tqdm

from IPython.display import display, Image
import matplotlib.pyplot as plt

'''
To-Do:
1. parallelize the data loading process for each participant-trial
2. 
'''

## the order of the 4 labels
label_names = ['valence', 'arousal', 'dominance', 'liking'] # 4 labels
## the order of the 32 EEG channels
eeg_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 
                'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 
                'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 
                'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'] # 32 channels
## the order of the 8 other physiological channels
other_channels = ['hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Resp', 'Pleth', 'Temp'] # 8 channels

def find_participant_ids(data_folder):
    participant_ids = list()
    for x in sorted(os.listdir(data_folder)):
        participant_id = x[1:]
        participant_data_folder = os.path.join(data_folder, x)
        if os.path.isdir(participant_data_folder):
            data_file = os.path.join(participant_data_folder, 's' + participant_id + '.mat')
            if os.path.isfile(data_file):
                participant_ids.append(participant_id)
    return sorted(participant_ids)


def extract_frame_per_5sec(cap, display=False):
    # Set the frame rate and interval for frame extraction
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    interval_seconds = 5
    interval_frames = frame_rate * interval_seconds

    # Initialize variables
    frame_count = 0

    frame_rgb_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Extract a frame every 'interval_frames' frames
        if frame_count % interval_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb_list.append(frame_rgb[np.newaxis, :])
            if display:
                ## Display the extracted frame
                # plt.imshow(frame_rgb, cmap = plt.cm.Spectral)
                display(Image(data=cv2.imencode('.jpg', frame)[1].tobytes()))
    
    return np.vstack(frame_rgb_list)


def load_participant_data(data_folder, participant_id, trial_id, task='bi_class', load_video=True):
    '''
    input:
        data_folder: str, path to the data folder that contains data for all participants
        participant_id: str, participant id, e.g. '01'
        trial_id: str, trial id for the participant, e.g. '01'
        task: str, task type, 'bi_class' or 'regression'; if binary classification, 
                the labels are binarized to high/versus for each emotion. 
                If label <= 5, then low (0); otherwise, high (1)
        load_video: bool, whether to load the video data
    return:
        a dictionary containing the labels, video, eeg, and other physiological signals
            labels: np.array, (4, )
            video: np.array, (num_frames, 576, 720, 3)
            eeg_data: np.array, (32, 8064)
            other_physio_data: np.array, (8, 8064)

    '''
    if load_video:
        ## video
        cap = cv2.VideoCapture('{0}/P{1}/s{1}/s{1}_trial{2}.avi'.format(data_folder, participant_id, trial_id)) # each frame: (576, 720, 3)
        if not cap.isOpened():
            print("\tError: Could not open video file for s{0}_trial{1}.".format(participant_id, trial_id))
            face_frame_list = None
        else:
            face_frame_list = extract_frame_per_5sec(cap)    
            # Release the video capture object and close the video file
            cap.release()
    else:
        face_frame_list = None
    
    mat = sp.io.loadmat( '{0}/P{1}/s{1}.mat'.format(data_folder, participant_id) )
    ## labels
    labels = mat['labels'][int(trial_id)-1, :] # video/trial x label (valence, arousal, dominance, liking)
    if task == 'bi_class':
        labels = np.array([1 if l > 5 else 0 for l in labels])
    ## physiological signals
    physio_data = mat['data'] # video/trial x channel x data (physiological signals)
    # EEG
    eeg_data = physio_data[int(trial_id)-1, :32, :]
    # other physiological signals
    other_physio_data = physio_data[int(trial_id)-1, 32:, :]

    return {'participant_id': participant_id,
            'trial_id': trial_id,
            'labels': labels,
            'video': face_frame_list,
            'eeg_data': eeg_data,
            'other_physio_data': other_physio_data}
