from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_pca_file="./result/matrix_pca.txt") :
        super().__init__()
        self.data_pca_file = data_pca_file
        self.length = np.loadtxt(self.data_pca_file).shape[0]
    
    def __getitem__(self, index):
        all_data = np.loadtxt(self.data_pca_file)
        return all_data[index, :]
    
    def __len__(self):
        return self.length