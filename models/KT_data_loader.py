from torch.utils.data import Dataset


class AAE_KTDATA(Dataset):
    def __init__(self, data):
        '''
        data: numpy array (N x D)
        '''
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        return sample