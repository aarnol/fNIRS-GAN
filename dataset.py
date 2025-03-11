import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import scipy.io
import pandas as pd
from scipy.stats import zscore
sampling_rate = 3.8147
def get_onset_index(timing, sampling_rate):
    timing = timing / sampling_rate
    return int(timing)
class FNIRSDataset(Dataset):
    def __init__(self, subject_ids, condition='nback', type='HbR', transform=None):
        """
        Args:
            subject_ids (list): List of subject IDs to load data for.
            condition (str): Experimental condition ('nback' or other).
            type (str): fNIRS signal type ('HbO', 'HbR', 'HbT').
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.data = []
        for subject_id in subject_ids:
            self.data.extend(self.load_fnirs_subject(subject_id, condition, type))
        # Pad the data so both dimensions are divisible by 4
        for i in range(len(self.data)):
            sample = self.data[i]['roiTimeseries']
            pad_height = (4 - sample.shape[0] % 4) % 4
            pad_width = (4 - sample.shape[1] % 4) % 4
            self.data[i]['roiTimeseries'] = np.pad(sample, ((0, pad_height), (0, pad_width)), mode='constant')
        self.transform = transform
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]['roiTimeseries']
        label = self.data[idx]['pheno']['label']
        
        if self.transform:
            sample = self.transform(sample)
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
    def load_fnirs_subject(self, subject_id, condition='nback', type='HbR'):
        type_dict = {'HbO': 0, 'HbR': 1, 'HbT': 2}
        target_folder = f"data"
        data_path = os.path.join(target_folder, 'fNIRS_HCP_SubjSpecific 1.mat')
        data = scipy.io.loadmat(data_path)['Data_fNIRS'][subject_id - 1][0]
        
        if condition == 'nback':
            data = data[:, :3]
        else:
            data = data[:, 4:]
        
        data = data[:, type_dict[type]]
        formatted_data = []
        labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]
        conditions = [4, 1, 2, 4, 1, 3, 2, 3, 4, 1, 2, 4, 1, 3, 2, 3]
        
        for i, block in enumerate(data):
            f_data = {
                'roiTimeseries': block,
                'pheno': {
                    'subjectId': f'S{subject_id}',
                    'encoding': None,
                    'label': labels[i],
                    'condition': conditions[i],
                    'modality': 'fNIRS'
                }
            }
            formatted_data.append(f_data)
        
        return formatted_data

# Example usage
def get_fnirs_dataloader(subject_ids, condition='nback', type='HbR', batch_size=32, shuffle=True, num_workers=2):
    dataset = FNIRSDataset(subject_ids, condition, type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class ONRData(Dataset):
    def __init__(self, label_dtype = 'float', transform=None):
        self.participants = [
            'P07', 'P10', 'P11', 'P13', 'P14', 'P16', 'P17', 'P18', 'P22', 'P23', 
            'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 
            'P34', 'P35', 'P36'
            ]
        self.label_dtype = label_dtype
    
        timings = []
        for subject in self.participants:
            path = os.path.join("ONR_data","data", "lsl", subject+"_lsl.tri")
            timing = pd.read_csv(path, sep=";", header=None, names=['time', 'onset', 'code'])
            timing['time'] = pd.to_datetime(timing['time']).dt.strftime('%H:%M:%S')
            timings.append(timing)
        self.timings = timings
        
        data = {}
        for subject in self.participants:
            path = os.path.join("ONR_data","data", "data_csvs", subject+".csv")
            fnir = pd.read_csv(path, sep=",", header=0)
            fnir = fnir.drop(columns=['Time'])
            
            

            data[subject] = fnir
         
        
        
        
        
        
        workload = pd.read_csv("ONR_data/data/load/workload.csv")
        workload['timestamp'] = workload['timestamp'].str.replace('.', ':')
        workload_dict = {}
        for i in range(len(self.participants)):
            workload_dict[self.participants[i]] = workload[workload['participant_number'] == int(self.participants[i][1:])].iloc[:, :]
        for i in range(len(self.participants)):
            matched_times = timings[i][timings[i]['time'].isin(workload_dict[self.participants[i]]['timestamp'])]
            workload_dict[self.participants[i]] = workload_dict[self.participants[i]].merge(matched_times, left_on='timestamp', right_on='time', how='left')
        
        #parameters 
        window_size = 4
        #collating data and labels
        
       
        X = []
        y = []
        groups = []
        channel_structure = pd.read_csv("ONR_data/channel_structure.csv").to_numpy()
       
        #iterate through the subjects
        for sub in self.participants:
            num_samples = 0
            sub_metadata = workload_dict[sub]
            sub_data = data[sub].to_numpy()
            
            
            for row in sub_metadata.itertuples():
                onset = row.onset
                label = row.load_label

                #Get the corresponding fnirs sample 
                index = get_onset_index(onset, sampling_rate)
                
                #Calculate the window
                fnirs_sample = sub_data[index-window_size+1:index+1]
                fnirs_sample_structure = np.zeros((9,8,4))
                
                for i in range(9):
                    for j in range(8):
                        for k in range(4):
                            channel = int(channel_structure[i,j]/2 -1)
                           
                            fnirs_sample_structure[i,j,k] = fnirs_sample[k][channel]
                
                if np.isnan(fnirs_sample).any():
                    continue
                
                
                X.append(fnirs_sample_structure)
                
                y.append(label)
                groups.append(sub)
            
            
        X = np.array(X)
        y = np.array(y)
        
        self.X = X
        
        self.y = y
        self.groups = groups
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        
        pad_height = (4 - sample.shape[0] % 4) % 4 
        pad_width = (4 - sample.shape[1] % 4) % 4 
        sample = torch.permute(sample, (2, 0, 1))
        #normalize across the channel dimension
        # Compute mean and std along the 0th dimension (D)
        mean = sample.mean(dim=0, keepdim=True)
        std = sample.std(dim=0, keepdim=True)

        # Normalize
        sample = (sample - mean) / (std + 1e-8)  # Adding epsilon for numerical stability

        sample = torch.nn.functional.pad(sample, (0, pad_width, 0, pad_height), mode='constant', value=0)
        
       
        
        
        if label == "optimal":
            label = 0
        else:
            label = 1
        if self.label_dtype == 'int':
            return sample.to(torch.float32), torch.tensor(label, dtype=torch.int64)
        else:
            return sample.to(torch.float32), torch.tensor(label, dtype=torch.float32)
