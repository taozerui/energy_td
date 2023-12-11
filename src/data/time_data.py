import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def get_air_data(file_path: str, batch_size=int(2**10)):

    data_loaders = []
    dt = np.load(os.path.join(file_path, 'beijing.npy'), allow_pickle=True)
    dt = dt.item()
    train_dt_raw = dt['train_folds']
    test_dt_raw = dt['test_folds']

    dt_save = []
    for i in range(5):
        train_ind = train_dt_raw[i][:, :2].astype(int)
        train_time = train_dt_raw[i][:, 2:3].astype(float).reshape(-1) / 10.
        train_val = train_dt_raw[i][:, -1].astype(float).reshape(-1) / 5.

        test_ind = test_dt_raw[i][:, :2].astype(int)
        test_time = test_dt_raw[i][:, 2:3].astype(float).reshape(-1) / 10.
        test_val = test_dt_raw[i][:, -1].astype(float).reshape(-1) / 5.

        train_ind = torch.tensor(train_ind, dtype=torch.int64)
        test_ind = torch.tensor(test_ind, dtype=torch.int64)

        train_val = torch.tensor(train_val, dtype=torch.float32)
        test_val = torch.tensor(test_val, dtype=torch.float32)

        train_time = torch.tensor(train_time, dtype=torch.float32)
        test_time = torch.tensor(test_time, dtype=torch.float32)

        train_dt = TensorDataset(train_ind, train_time, train_val)
        test_dt = TensorDataset(test_ind, test_time, test_val)

        train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dt, batch_size=100, shuffle=False)

        data_loaders.append(
            {'train': train_loader, 'test': test_loader}
        )
        dt_save.append(
            {'train_ind': train_ind, 'train_val': train_val, 'train_time': train_time,
             'test_ind': test_ind, 'test_val': test_val, 'test_time': test_time,}
        )
    # with open('beijing_air.pkl', 'wb') as f:
    #     pickle.dump(dt_save, f)
    # import ipdb; ipdb.set_trace()

    return data_loaders
