data_folder = 'drive/MyDrive/DEAP'
task = 'regression'
import pandas as pd
import json
import numpy as np

from helper_code import load_participant_data, find_participant_ids
from helper_code import eeg_channels as EEG_CHANNELS
from helper_code import label_names as LABEL_NAMES

data_folder = data_folder[:-1] if data_folder.endswith('/') else data_folder
participant_ids = find_participant_ids('drive/MyDrive/DEAP/')
print('participant_ids', participant_ids)
video_features = []
eeg_features = []
other_physio_features = list()
labels_all = list()
print('loading data...')
f = open("eeg_encoded_data.txt", "r")
facial_embedding_data = f.read()
facial_embedding_data = json.loads(facial_embedding_data)

X, y = [], []
for participant_id in participant_ids[:16]:
    try:
      trials_len = 40

      for trial_id in range(1, trials_len):
          trial_id_mod = '0' + str(trial_id) if trial_id < 10 else str(trial_id)
          #participant_id_mod = '0' + str(participant_id) if participant_id < 10 else str(participant_id)

          labels = load_participant_data(data_folder, participant_id, trial_id_mod)['labels']
          #participant_id = '0' + str(participant_id) if participant_id < 10 else str(participant_id)
          url = '{0}/s{1}_trial{2}.pkl'.format(data_folder + '/Video_embeddings', participant_id, trial_id_mod)
          video_embedding = pd.read_pickle(url)
          
          participant_id_int = int(participant_id) if participant_id[0] != '0' else int(participant_id[1:])
          face_embedding = np.array(facial_embedding_data[str(participant_id_int)][str(trial_id)])

          feat_vec = np.concatenate((video_embedding.detach().cpu().numpy().flatten(), face_embedding.flatten()), axis = 0)
          print(feat_vec.shape)
          if feat_vec.shape[0] == 2409226:
            X.append(feat_vec)
            y.append(labels)

    except Exception as e:
      print(e)



np.savetxt('features.txt', np.array(X) )    # .npy extension is added if not given
np.savetxt('labels.txt', np.array(y)  )  # .npy extension is added if not given