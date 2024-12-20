import json
import os
import jieba


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