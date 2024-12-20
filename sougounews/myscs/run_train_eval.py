import torch
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn as nn
import jieba
import torch.optim as optim
from tqdm import tqdm
from functools import partial
import config as cg
import vocab 
import model as md
import utils 

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

def collate_fn(batch, word2idx):
    texts, labels = zip(*batch)
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=word2idx['<PAD>'])
    labels = torch.tensor(labels)
    return texts, labels



def train_model(config, train_loader, val_loader):
    model = md.DPCNN(config).to(config.device)
    criterion = (utils.LabelSmoothingLoss(num_classes=config.num_classes, smoothing = config.smoothing) if config.isLabelSmoothingLoss else  nn.CrossEntropyLoss())
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.iswarmup:
        total_steps = config.num_epochs * len(train_loader)  
        warmup_steps = total_steps // 10 
        scheduler = utils.get_warmup_scheduler(optimizer, warmup_steps, total_steps)


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
            if(config.iswarmup):
                scheduler.step()

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

    # 如果词表文件存在，加载词表；否则重新构建并保存
    config = cg.Config()
    if os.path.exists(config.vocab_path):
        word2idx, idx2word = vocab.load_vocab(config.vocab_path)
    else:
        word2idx, idx2word = vocab.build_vocab(config.data_path)
        vocab.save_vocab(word2idx, idx2word, config.vocab_path)

    config.n_vocab = len(word2idx)

    dataset = NewDataset(config.data_path, word2idx)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    collate_fn_with_vocab = partial(collate_fn, word2idx=word2idx)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_with_vocab)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_with_vocab)

    train_model(config, train_loader, val_loader)