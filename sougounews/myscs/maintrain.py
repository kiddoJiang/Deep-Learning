import torch
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn as nn
import jieba
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from functools import partial
import json



def save_vocab(word2idx, idx2word, vocab_path):
    vocab = {'word2idx': word2idx, 'idx2word': idx2word}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"词表已保存到 {vocab_path}")

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"词表已加载自 {vocab_path}")
    return vocab['word2idx'], vocab['idx2word']

class NewDataset(Dataset):
    def __init__(self, data_path, word2idx, max_len=100):
        self.data_path = data_path
        self.word2idx = word2idx
        self.max_len = max_len
        self.texts = []
        self.labels = []

        for label, label_name in enumerate(os.listdir(data_path)):
            label_dir = os.path.join(data_path, label_name)
            print(f"加载类别：{label_name}, 文件数：{len(os.listdir(label_dir))}")
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text = self.text_process(text)
                    self.texts.append(text)
                    self.labels.append(label)

    def text_process(self, text):
        if not text.strip():
            return ['<PAD>'] * self.max_len
        text = jieba.cut(text)
        text = list(text)
        if len(text) > self.max_len:
            text = text[:self.max_len]
        else:
            text = text + ['<PAD>'] * (self.max_len - len(text))
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_idx = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text]
        return torch.tensor(text_idx), torch.tensor(label)

def build_vocab(data_path, max_vocab_size=10000):
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx2word = {0: '<PAD>', 1: '<UNK>'}
    word_count = {}

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

class Config:
    def __init__(self, dataset):
        self.model_name = 'DPCNN'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.require_improvement = 1000
        self.class_list = os.listdir(dataset)  # 自动获取类别列表
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = 300  # 字向量维度
        self.num_filters = 250

def collate_fn(batch, word2idx):
    texts, labels = zip(*batch)
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=word2idx['<PAD>'])
    labels = torch.tensor(labels)
    return texts, labels

class DPCNN(nn.Module):
    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv_region(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()
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
        x = x + px
        return x

def train_model(config, train_loader, val_loader):
    model = DPCNN(config).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        model.train()
        train_loss, train_accs = [], []

        for texts, labels in tqdm(train_loader):
            texts, labels = texts.to(config.device), labels.to(config.device)
            output = model(texts)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc.item())

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[训练 | {epoch + 1}/{config.num_epochs}] 损失 = {train_loss:.4f}, 准确率 = {train_acc:.4f}")

        model.eval()
        val_loss, val_accs = [], []
        with torch.no_grad():
            for texts, labels in tqdm(val_loader):
                texts, labels = texts.to(config.device), labels.to(config.device)
                output = model(texts)
                loss = criterion(output, labels)
                acc = (output.argmax(dim=-1) == labels).float().mean()
                val_loss.append(loss.item())
                val_accs.append(acc.item())

        val_loss = sum(val_loss) / len(val_loss)
        val_acc = sum(val_accs) / len(val_accs)
        print(f"[验证 | {epoch + 1}/{config.num_epochs}] 损失 = {val_loss:.4f}, 准确率 = {val_acc:.4f}")

if __name__ == "__main__":
    data_path = 'sougounews/SougoCS_all_classes'
    vocab_path = 'sougounews/vocab.json'

    # 如果词表文件存在，加载词表；否则重新构建并保存
    if os.path.exists(vocab_path):
        word2idx, idx2word = load_vocab(vocab_path)
    else:
        word2idx, idx2word = build_vocab(data_path)
        save_vocab(word2idx, idx2word, vocab_path)

    config = Config(data_path)
    config.n_vocab = len(word2idx)

    dataset = NewDataset(data_path, word2idx)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    collate_fn_with_vocab = partial(collate_fn, word2idx=word2idx)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_with_vocab)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_with_vocab)

    train_model(config, train_loader, val_loader)
