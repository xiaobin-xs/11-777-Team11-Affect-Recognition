import scipy as sp
import numpy as np
import os
import mne
from helper_code import (eeg_channels,other_channels, label_names)
import matplotlib.pyplot as plt


def load_phyis_data(path,trial):
    mat = sp.io.loadmat(path)
    physio_data = mat['data'][int(trial)-1]
     # EEG
    eeg_data = physio_data[:32, :]
    # other physiological signals
    other_physio_data = physio_data[32:, :]
    return {
        'EEG': eeg_data,
        'other': other_physio_data,
    }

def mne_plot(eeg_data, save_path):
    mne_info = mne.create_info(eeg_channels, 128, ch_types='eeg', verbose=False)
    # evoked = mne.EvokedArray(eeg_data, mne_info, verbose= True)
    
    raw = mne.io.RawArray(eeg_data, mne_info, verbose= False)
    raw.set_montage('standard_1020')
    topo_map = raw.plot_psd_topomap(dB=True, show = False)
    topo_map.savefig('prakarsh/plots/'+save_path+'_eeg', bbox_inches='tight',dpi = 150)
    

def phys_plot(data, save_path):
    
    fig, axes = plt.subplots(8,1)
    fig.set_size_inches(5,10)
    axes = axes.ravel()
    xaxis= np.linspace(0,data.shape[1]/128,data.shape[1])
    for i, ax in enumerate(axes):
        ax.set_ylabel(other_channels[i])
        ax.plot(xaxis, data[i,:])
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig('prakarsh/plots/'+save_path+'_other', bbox_inches='tight',dpi = 150)
    
def human_class(args):
    participants_use = [('P01','10'),('P01','23'),('P02','02'),('P02','05'),('P02','19'),('P02','39'),('P04','23'),
                        ('P09','24'),('P11','20'),('P13','14'),('P13','26'),('P18','40'),('P22','33'),('P23','35'),
                        ('P25','34'),('P26','16'),('P27','14'),('P27','27'),('P27','33'),('P29','21'),('P29','31'),
                        ]

    data_dict = {}
    for participant in participants_use:
        data_dict [participant[0]+'_'+participant[1]]= load_phyis_data(os.path.join(args.data_dir,participant[0],'s'+participant[0][1:]), participant[1])
    
    for trial in data_dict:
        mne_plot(data_dict[trial]['EEG'],trial)
        phys_plot(data_dict[trial]['other'],trial)

    return None
