import os
from gensim.models.word2vec import Word2Vec
import pickle


class SplitVec7:
    def split_text(self, file="poetry_7.txt", train_num=6000):
        all_data = open(file, "r", encoding="utf-8").read()
        with open("split_7.txt", "w", encoding="utf-8") as f:
            split_data = " ".join(all_data)
            f.write(split_data)
        return split_data[:train_num * 64]

    def train_vec(self, vector_size=128, split_file="split_7.txt", org_file="poetry_7.txt", train_num=6000):
        param_file = "word_vec7.pkl"
        org_data = open(org_file, "r", encoding="utf-8").read().split("\n")[:train_num]
        if os.path.exists(split_file):
            all_data_split = open(split_file, "r", encoding="utf-8").read().split("\n")[:train_num]
        else:
            all_data_split = self.split_text().split("\n")[:train_num]

        if os.path.exists(param_file):
            return org_data, pickle.load(open(param_file, "rb"))

        models = Word2Vec(all_data_split, vector_size=vector_size, workers=7, min_count=1)
        pickle.dump([models.syn1neg, models.wv.key_to_index, models.wv.index_to_key], open(param_file, "wb"))
        return org_data, (models.syn1neg, models.wv.key_to_index, models.wv.index_to_key)
