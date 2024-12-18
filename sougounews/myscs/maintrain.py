import torch
from torch.utils.data import DataLoader,Dataset
import os
import torch.nn as nn
import jieba
import torch.nn.functional as F
import numpy as np

class newdataset (Dataset):
    def __init__(self,data_path,label_dict,max_len = 100):
        self.data_path = data_path
        self.label_dict = label_dict
        self.max_len = max_len
        self.texts = []
        self.labels = []

        for label, label_name in enumerate(os.listdir(data_path)):
            label_dir = os.path.join(data_path,label_dir)
            for file_name in os.listdir(label_dir):
                label_path = os.path.join(label_dir,file_name)
                with open(label_path,'r',encoding='utf_8') as f:
                    text = f.read()
                    text = self.text_process(text)
                    self.texts.append(text)
                    self.labels.append(label)

    def text_process(self, text):
        text = jieba.cut(text)
        text = list(text)
        if len(text) > self.max_len:
            text = text[:self.max_len]
        else:
            text = text + ['<PAD>'] * (self.max_len - len(text))
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_idx = [word2idx.get(word, word2idx['<UNK>']) for word in text]
        return torch.tensor(text_idx), torch.tensor(label)

def build_vocab(data_path, max_vocab_size=10000):
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx2word = {0: '<PAD>', 1: '<UNK>'}
    word_count = {}
    
    # 统计词频
    for label_name in os.listdir(data_path):
        label_dir = os.path.join(data_path, label_name)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text = jieba.cut(text)
                for word in text:
                    word_count[word] = word_count.get(word, 0) + 1

    sorted_vocab = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    for i, (word, _) in enumerate(sorted_vocab[:max_vocab_size - 2]):
        word2idx[word] = i + 2
        idx2word[i + 2] = word
    
    return word2idx, idx2word

class Config():

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'DPCNN'
        # self.train_path = dataset + '/data/train.txt'                                # 训练集
        # self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        # self.test_path = dataset + '/data/test.txt'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        # self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        # self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        # self.log_path = dataset + '/log/' + self.model_name
        # self.embedding_pretrained = torch.tensor(
        #     np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
        #     if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.num_filters = 250                                          # 卷积核数量(channels数)

def collate_fn(batch):

    texts, labels = zip(*batch)
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=word2idx['<PAD>'])
    labels = torch.tensor(labels)
    
    return texts, labels

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
