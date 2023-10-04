from helper_code import load_participant_data

if __name__ == '__main__':
    data_folder = 'DEAP/data'
    participant_id = '01'
    trial_id = '01'
    data_folder = data_folder[:-1] if data_folder.endswith('/') else data_folder
    participant_id = participant_id.zfill(2)
    trial_id = trial_id.zfill(2)
    data_dict = load_participant_data(data_folder, participant_id, trial_id, task='bi_class')
