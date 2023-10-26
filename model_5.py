import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from split_vec5 import SplitVec5
from poetry_dataset import PoetryDataset7

from pypinyin import lazy_pinyin, Style


class PoetryModellstm5(nn.Module):
    def __init__(self, params):
        """
        all_data:用列表储存的诗歌数据集
        w1:经过Word2vec模型训练得到的词向量矩阵
        word_2_index:字典，词到索引的映射
        index_2_word:列表，索引到词的映射
        word_size:词向量矩阵中词的数量
        embedding_num: 词向量的维度
        :param params:
        """
        super().__init__()
        self.all_data, (self.w1, self.word_2_index, self.index_2_word) = SplitVec5().train_vec(
            vector_size=params["embedding_num"],
            train_num=params["train_num"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hidden_num = params["hidden_num"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.optimizer = params["optimizer"]
        self.word_size, self.embedding_num = self.w1.shape

        # 定义一个LSTM层并将其赋值给'self.lstm'。
        # 输入大小为词向量的维度，隐藏层的大小由之前的'hidden_num'决定，批次大小为True（这是LSTM的默认批次大小设置），层数为2，并且是单向的。
        # 因为双向的话网络会考虑前后语境的信息，对于有明确顺序的文本如小说文章有较好的效果。但对于诗歌这种无明确语境顺序的文本，单向有着较好效果。
        self.lstm = nn.LSTM(input_size=self.embedding_num, hidden_size=self.hidden_num, batch_first=True, num_layers=2,
                            bidirectional=False)

        self.dropout = nn.Dropout(0.3)  # 古诗不具有唯一性
        self.flatten = nn.Flatten(0, 1)  # Flatten层将输入张量展平,以便可以输入到全连接层
        # 定义一个全连接层并将其赋值给'self.linear'。这个全连接层的输入大小为隐藏层的单元数量，输出大小为词的数量（即词向量矩阵的词数）
        self.linear = nn.Linear(self.hidden_num, self.word_size)
        self.cross_entropy = nn.CrossEntropyLoss()  # 选择交叉熵为损失函数

    def forward(self, xs_embedding, h_0=None, c_0=None):
        """
        :param xs_embedding: 输入的词向量
        :param h_0: LSTM模型的隐藏状态
        :param c_0: LSTM模型的细胞状态
        :return:
        """
        if h_0 is None or c_0 is None:
            h_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            c_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        xs_embedding = xs_embedding.to(self.device)
        hidden, (h_0, c_0) = self.lstm(xs_embedding, (h_0, c_0))
        hidden_drop = self.dropout(hidden)
        hidden_flatten = self.flatten(hidden_drop)
        pre = self.linear(hidden_flatten)

        return pre, (h_0, c_0)

    def to_train(self):
        model_result_file = "Poetry_Model_lstm_model_5.pkl"
        # 训练完成则直接调用模型
        if os.path.exists(model_result_file):
            return pickle.load(open(model_result_file, "rb"))

        # 创建了一个名为'PoetryDataset7'的诗歌数据集
        dataset = PoetryDataset7(self.w1, self.word_2_index, self.all_data)
        # 使用数据集和批次大小创建了一个数据加载器，用于逐批次加载数据。
        dataloader = DataLoader(dataset, self.batch_size)

        optimizer = self.optimizer(self.parameters(), self.lr)  # 根据给定的参数和learning rate创建了一个优化器。
        self = self.to(self.device)  # 将模型移动到GPU上
        for e in range(self.epochs):
            # 在每个训练周期中，对数据加载器逐批次加载数据。
            # batch_index训练批次数, batch_x_enbedding是每个批次古诗的三维矩阵（该批次的古诗数量，每首古诗的字数(不算句号)，每个字的词向量）
            # batch_y_index二维矩阵（每批次的古诗数量，每首古诗的每个字在index_2_word中的索引序号)
            for batch_index, (batch_x_embedding, batch_y_index) in enumerate(dataloader):
                self.train()
                batch_x_embedding = batch_x_embedding.to(self.device)
                batch_y_index = batch_y_index.to(self.device)

                pre, _ = self.forward(batch_x_embedding)  # 对输入数据进行前向传播。
                loss = self.cross_entropy(pre, batch_y_index.reshape(-1))  # 计算损失，使用了交叉熵损失函数。
                loss.backward()  # 通过反向传播算法计算梯度 , 梯度累加, 但梯度并不更新, 梯度是由优化器更新的
                optimizer.step()  # 使用优化器更新梯度
                optimizer.zero_grad()  # 梯度清零

                if batch_index % 100 == 0:
                    print(f"loss:{loss:.3f}")
                    self.generate_poetry_auto()
        pickle.dump(self, open(model_result_file, "wb"))  # 将训练好的模型保存到文件中。
        return self  # 返回训练好的模型

    def generate_poetry_auto(self):
        # self.eval()
        result = ""
        # 生成一个从0到self.word_size（单词数量）的随机整数作为当前选择的单词的索引
        word_index = np.random.randint(0, self.word_size, 1)[0]

        result += self.index_2_word[word_index]  # 获取对应汉字
        # 初始化一个大小为2x1xself.hidden_num的全零tensor，并将其存储在GPU上
        h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32), device='cuda')
        # 初始化一个大小为2x1xself.hidden_num的全零tensor，并将其存储在GPU上
        c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32), device='cuda')

        for i in range(23):
            # 从w1中获取当前单词的嵌入向量并将其转化为一个tensor，然后将其移动到GPU上
            word_embedding = torch.tensor(self.w1[word_index][None][None], device='cuda')
            self.to('cuda')
            # 对当前的单词嵌入、隐藏状态和细胞状态进行前向传播，得到预测结果和新的隐藏状态与细胞状态。
            pre, (h_0, c_0) = self.forward(word_embedding, h_0, c_0)
            # 对预测结果进行argmax操作，获取最大概率对应的单词索引，然后将其转换为整数。
            word_index = int(torch.argmax(pre))
            # 添加到result
            result += self.index_2_word[word_index]
        if self.count_punctuation(result):
            if self.cut(result):
                return result
            else:
                temp = self.generate_poetry_auto()
                return temp
        else:
            temp = self.generate_poetry_auto()
            return temp

    def generate_poetry_acrostic(self, str_in):

        input_text = str_in[:len(str_in)]
        if input_text == "":
            self.generate_poetry_auto()
        else:
            result = ""
            punctuation_list = ["，", "。"]
            for num in range(len(input_text) // 2):
                punctuation_list += punctuation_list
            for i in range(len(input_text)):
                h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32), device='cuda')
                c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32), device='cuda')
                word = input_text[i]  # 获取输入的文本中的第i个字
                try:
                    word_index = self.word_2_index[word]  # 尝试获取字'word'的索引。
                except:
                    # 如果字'word'不在word_2_index中，那么随机生成一个索引并使用它来从index_2_word中获取一个随机字。
                    word_index = np.random.randint(0, self.word_size, 1)[0]
                    word = self.index_2_word[word_index]
                result += word

                for j in range(4):
                    word_index = self.word_2_index[word]
                    # 获取当前字的索引并使用它从w1中获取字的嵌入向量。
                    word_embedding = torch.tensor(self.w1[word_index][None][None])
                    self.to('cuda')
                    # 对字的嵌入进行前向传播，得到预测结果和新的隐藏状态与细胞状态。
                    pre, (h_0, c_0) = self.forward(word_embedding, h_0, c_0)
                    # 使用预测结果选择一个新字并添加到结果字符串中。
                    word = self.index_2_word[int(torch.argmax(pre))]
                    result += word
                result += punctuation_list[i]
            if self.cut(result):
                return result
            else:
                temp = self.generate_poetry_acrostic(str_in)
                return temp

    # 判断是否押韵
    def cut(self, result):
        temp = []
        for i in range(len(result)):
            if result[i] == '。':
                temp.append(lazy_pinyin(result[i - 1], style=Style.FINALS)[0])
        if len(temp) != len(set(temp)):
            return True
        else:
            return False

    # 判断诗歌生成时候的标点符号是否出错
    def count_punctuation(self, poem):
        comma_count = poem.count('，')
        period_count = poem.count('。')
        if comma_count == 2 and period_count == 2:
            return True
        else:
            return False


if __name__ == "__main__":
    # ---------------------------------  个性化参数  --------------------------------------
    params = {
        "batch_size": 32,  # batch(古诗数)大小
        "epochs": 2000,  # epoch大小
        "lr": 0.001,  # 学习率
        "hidden_num": 64,  # 隐层大小
        "embedding_num": 128,  # 词向量维度
        "train_num": 2000,  # 训练的古诗数量, 七言古诗:0~6290, 五言古诗:0~2929
        "optimizer": torch.optim.AdamW,  # 优化器 , 注意不要加括号
        "batch_num_test": 100,  # 多少个batch 打印一首古诗进行效果测试
    }

    model = PoetryModellstm5(params)  # 模型定义
    model = model.to_train()  # 模型训练