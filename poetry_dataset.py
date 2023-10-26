from torch.utils.data import Dataset
import numpy as np


class PoetryDataset7(Dataset):
    """
    __init__是类的构造函数，接受三个参数：w1，word_2_index和all_data。

    w1：是一个词嵌入矩阵，用于将单词映射到一组数字（向量）。词嵌入是一种将词语映射到实数向量的方法，常用于自然语言处理任务。
    word_2_index：这是一个字典，将单词映射为索引。这通常是预处理数据的一部分，以便在创建训练样本时将单词转换为数字。
    all_data：这可能是一个包含所有诗歌数据的列表。
    __getitem__方法用于获取数据集中的单个样本。它接受一个索引（index）作为输入，并返回一个元组。这个元组包含两个元素：
    xs_embedding：这是一个嵌入向量，由输入诗歌（a_poetry）的每个单词的索引查找词嵌入矩阵w1得出。
    注意，这里只考虑了输入诗歌中每个单词的前一个单词的嵌入向量（xs是a_poetry_index[:-1]，即除了最后一个单词之外的所有单词的索引）。
    ys：这是一个数组，包含输入诗歌中每个字的下一个字的索引（ys = a_poetry_index[1:]）。这个数组被转换为整数类型（通过np.array(ys).astype(np.int64)）。
    """

    def __init__(self, w1, word_2_index, all_data):
        self.w1 = w1
        self.word_2_index = word_2_index
        self.all_data = all_data

    def __getitem__(self, index):
        # 获取单首诗歌
        a_poetry = self.all_data[index]

        # 获取该诗中每个字的索引
        a_poetry_index = [self.word_2_index[i] for i in a_poetry]

        xs = a_poetry_index[:-1]
        # ys只需要索引是因为计算损失的时候，目标诗歌只需要索引就可以
        ys = a_poetry_index[1:]

        xs_embedding = self.w1[xs]

        return xs_embedding, np.array(ys).astype(np.int64)

    def __len__(self):
        return len(self.all_data)

