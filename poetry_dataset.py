from torch.utils.data import Dataset
import numpy as np


class PoetryDataset7(Dataset):
    def __init__(self, w1, word_2_index, all_data):
        self.w1 = w1
        self.word_2_index = word_2_index
        self.all_data = all_data

    def __getitem__(self, index):
        a_poetry = self.all_data[index]

        a_poetry_index = [self.word_2_index[i] for i in a_poetry]
        xs = a_poetry_index[:-1]
        ys = a_poetry_index[1:]
        xs_embedding = self.w1[xs]

        return xs_embedding, np.array(ys).astype(np.int64)

    def __len__(self):
        return len(self.all_data)