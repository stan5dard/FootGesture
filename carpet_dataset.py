from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import torch

def remove_zero(data):
    idxs = []
    for i in range(len(data)):
        if data[i].mean() == 0.0:
            idxs.append(i)
    data = np.delete(data, idxs, 0)
    return data

class NumDataset(Dataset):

    def __init__(self, args):

        i = 0
        n = 0
        self.window_size = args.window_size
        results = []
        self.results2 = []
        self.labels = []
        print('hdf5 files preprocessing... start')
        for labelname in sorted(os.listdir(args.path)):
            for filename in sorted(os.listdir(args.path + labelname + '/')):
                print(n+1, labelname,'completed !')
                n = n + 1

                with h5py.File(args.path + labelname + '/' + filename, "r") as f:
                    data = np.array(list(f["pressure"]))
                    data = remove_zero(data)
                    length = len(data)

                    data = data.reshape(length, -1)

                    for a in range(length - args.window_size):
                        self.results2.append(data[a:a + args.window_size, :])
                        self.labels.append(i)

                results.append(data)

            i = i + 1

        self.results2 = torch.tensor(self.results2)
        print('hdf5 files preprocessing... end')
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        num_of_idx = len(self.results2) - self.window_size
        return num_of_idx

    def __getitem__(self, idx):
        idx = idx + 1
        X = self.results2[idx]
        y = self.labels[idx]

        return X, y