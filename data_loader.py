
#class CustomDataset(Dataset):
#   def __init__(self, args):
#        self.args = args
#        self.load_data()
#        self.torch_form()

#    def load_data(self):
#        s = self.args.train_subject[0]
#        if self.args.phase == 'train':
#            self.X = np.load(f"./data/S{s:02}_train_X.npy")
#            self.y = np.load(f"./data/S{s:02}_train_y.npy")
#        else:
#            self.X = np.load(f"./data/S{s:02}_test_X.npy")
#            self.y = np.load(f"./answer/S{s:02}_y_test.npy")
#        if len(self.X.shape) <= 3:
#            self.X = np.expand_dims(self.X, axis=1)

#    def torch_form(self):
#        self.X = torch.FloatTensor(self.X)
#        self.y = torch.LongTensor(self.y)

#    def __len__(self):
#        return len(self.X)

#    def __getitem__(self, idx):
#        sample = [self.X[idx], self.y[idx]]
#        return sample

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, source_list):
        self.sources = source_list # ex. if test_subject =1, then list : [2,3,4,5,6,7,8,9]
        self.n_ch =22 #BCI2a, number of EEG channels
        self.n_time = 1125 # BCI2a, number of timpoints

        self.load_data()
        self.torch_form()


    def load_data(self):
        #Build empty matrix ( for stacking EEG vectors of all subjects)
        self.X = np.empty(shape=(0, 1, self.n_ch, self.n_time), dtype=np.float32)
        self.y = np.empty(shape=(0), dtype=np.int32)

        for s in self.sources:
            Xtr = np.load(f"./data/S{s:02}_train_X.npy")
            ytr = np.load(f"./data/S{s:02}_train_y.npy")

            Xts = np.load(f"./data/S{s:02}_test_X.npy")
            yts = np.load(f"./answer/S{s:02}_y_test.npy")

            if len(Xtr.shape) <= 3:
                Xtr = np.expand_dims(Xtr, axis=1)
            if len(Xts.shape) <= 3:
                Xtr = np.expand_dims(Xtr, axis=1)

            X_tmp = np.concatenate((Xtr,  Xts), axis=0)
            y_tmp = np.concatenate((ytr,  yts), axis=0)

            self.X = np.concatenate((self.X, X_tmp), axis=0)
            self.y = np.concatenate((self.y, y_tmp), axis=0)  # 2~9 concat


    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = [self.X[idx], self.y[idx]]
        return sample



def data_loader(args):
    print("[Load data]")
    # Load train data
    args.phase = "train"
    dataset = CustomDataset(source_list=[2,3,4,5,6,7,8,9])
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size

    trainset, valset = random_split(dataset,[train_size, val_size])

    print(f"Training Data Size : {len(trainset)}")
    print(f"Testing Data Size : {len(valset)}")

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    # Load val data
    args.phase = "val"
    valset = CustomDataset(args) #shuffle =True
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    # Print
    print(f"train_set size: {train_loader.dataset.X.shape}")
    print(f"val_set size: {val_loader.dataset.X.shape}")
    print("")
    return train_loader, val_loader
