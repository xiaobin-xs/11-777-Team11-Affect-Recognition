import scipy as sp
import numpy as np
import cv2


from IPython.display import display, Image
import matplotlib.pyplot as plt

## the order of the 32 EEG channels
eeg_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 
                'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 
                'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 
                'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'] # 32 channels
## the order of the 8 other physiological channels
other_channels = ['hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Resp', 'Pleth', 'Temp'] # 8 channels


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
            frame_rgb_list.append(frame_rgb)
            if display:
                ## Display the extracted frame
                # plt.imshow(frame_rgb, cmap = plt.cm.Spectral)
                display(Image(data=cv2.imencode('.jpg', frame)[1].tobytes()))
    
    return frame_rgb_list


def load_participant_data(data_folder, participant_id, trial_id, task='bi_class'):
    '''
    input:
        data_folder: str, path to the data folder that contains data for all participants
        participant_id: str, participant id, e.g. '01'
        trial_id: str, trial id for the participant, e.g. '01'
        task: str, task type, 'bi_class' or 'regression'; if binary classification, 
                the labels are binarized to high/versus for each emotion. 
                If label <= 5, then low (0); otherwise, high (1)
    return:
        a dictionary containing the labels, video, eeg, and other physiological signals
            labels: np.array, 
    '''
    ## video
    cap = cv2.VideoCapture('{0}/P{1}/s{1}/s{1}_trial{2}.avi'.format(data_folder, participant_id, trial_id)) # each frame: (576, 720, 3)
    if not cap.isOpened():
        print("Error: Could not open video file for s{0}_trial{1}.".format(participant_id, trial_id))
        face_frame_list = None
    else:
        face_frame_list = extract_frame_per_5sec(cap)    
        # Release the video capture object and close the video file
        cap.release()
    
    mat = sp.io.loadmat( '{0}/P{1}/s{1}.mat'.format(data_folder, participant_id) )
    ## labels
    labels = mat['labels'][int(trial_id), :] # video/trial x label (valence, arousal, dominance, liking)
    if task == 'bi_class':
        labels = np.array([1 if l > 5 else 0 for l in labels])
    ## physiological signals
    physio_data = mat['data'] # video/trial x channel x data (physiological signals)
    # EEG
    eeg_data = physio_data[int(trial_id), :32, :]
    # other physiological signals
    other_physio_data = physio_data[int(trial_id), 32:, :]

    return {'participant_id': participant_id,
            'trial_id': trial_id,
            'labels': labels,
            'video': face_frame_list,
            'eeg_data': eeg_data,
            'other_physio_data': other_physio_data}
