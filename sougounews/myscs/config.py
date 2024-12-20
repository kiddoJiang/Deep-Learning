import torch
import os


class Config:
    def __init__(self):
        self.model_name = 'DPCNN'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = 'sougounews/SougoCS_all_classes'
        self.vocab_path = 'sougounews/vocab.json'
        self.dropout = 0.5
        self.class_list = os.listdir(self.data_path)  # 自动获取类别列表
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 10
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = 300  # 字向量维度
        self.num_filters = 250
        self.smoothing = 0.1
        self.iswarmup = False
        self.isLabelSmoothingLoss = True


