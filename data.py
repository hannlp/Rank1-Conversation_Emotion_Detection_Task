import csv
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len)
        self.with_label = True if labels is not None else False
        if self.with_label:
            self.labels = list(labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        if self.with_label:
            item['labels'] = torch.tensor(int(self.labels[index]))
        return item

    def __len__(self):
        return len(self.encodings.encodings) # 这个很坑，长度一定要看好

def split_and_load_dataset(data, tokenizer, max_len, batch_size, test_size, with_label=True, shuf=True):
    if with_label == False:
        test_dataset = TextClassificationDataset(data["Text"], None, tokenizer, max_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuf)
        return test_dataset, test_loader

    if test_size == 1.0:
        test_dataset = TextClassificationDataset(data["Text"], data["Labels"], tokenizer, max_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuf)
        print(len(test_loader))
        return test_dataset, test_loader

    X_train, X_val, y_train, y_val = train_test_split(data["Text"], data["Labels"], test_size=test_size, random_state=1)
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = TextClassificationDataset(X_val, y_val, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuf)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuf)
    return train_dataset, val_dataset, train_loader, val_loader

def process_train(file_path, shuf=True):
    dataset = list()
    csv_reader = csv.reader(open(file_path))
    for row in csv_reader:
        if row[2]:  # 标签存在
            strs = row[1].split('__eou__')#[:-1] # [:-1]可以去掉
            labels = list(str(row[2]))#[:-1] # 同
            for data in zip(strs, labels):
                dataset.append(data)

    data = pd.DataFrame(dataset, columns=['Text', 'Labels'])[1:]
    data['Labels'] = data['Labels'].astype(int)
    n_labels = len(set(data.Labels))

    cnt = Counter(list(data.Labels))

    label2idx = dict()
    idx2label = dict()
    for idx, label in enumerate(set(data.Labels)):
        label2idx[label] = idx
        idx2label[idx] = label
    for x in range(n_labels):
        assert label2idx[idx2label[x]] == x

    data['Labels'] = data.Labels.map(lambda x: label2idx[x])

    if shuf:
        data = shuffle(data)
    #data = data[:8000]
    return data, n_labels, cnt

def process_test(file_path, with_label=True, shuf=False):
    dataset = list()
    csv_reader = csv.reader(open(file_path))
    for row in csv_reader:
        if row[2]: # 标签存在
            strs = row[1].split('__eou__')
            if with_label:
                labels = list(str(row[2]))
                dataset.append([strs[-1], labels[-1]])
            else:
                dataset.append(strs[-1])

    if with_label:
        data = pd.DataFrame(dataset, columns=['Text', 'Labels'])[1:]
        data['Labels'] = data['Labels'].astype(int)
        n_labels = len(set(data.Labels))
        cnt = Counter(list(data.Labels))

        label2idx, idx2label = dict(), dict()
        for idx, label in enumerate(set(data.Labels)):
            label2idx[label] = idx
            idx2label[idx] = label
        for x in range(n_labels):
            assert label2idx[idx2label[x]] == x

        data['Labels'] = data.Labels.map(lambda x: label2idx[x])
        if shuf:
            data = shuffle(data)
        return data, n_labels, cnt, idx2label
    else:
        data = pd.DataFrame(dataset, columns=['Text'])[1:]
        if shuf:
            data = shuffle(data)
        return data


def make_batch(batch, device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device) if 'labels' in batch else None
    return input_ids, attention_mask, labels