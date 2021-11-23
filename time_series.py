from sklearn.metrics import classification_report
import os
import csv
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncode(nn.Module):
    def __init__(self, d_model, max_seq_len=512) -> None:
        super().__init__()
        self.pos_encode = self._get_pos_encode(max_seq_len, d_model)

    def forward(self, x):
        # - x: (batch_size, seq_len, d_model)
        return x + self.pos_encode[:x.size(1), :].unsqueeze(0).to(x.device)

    def _get_pos_encode(self, max_seq_len, d_model):
        pos_encode = torch.tensor([[pos / 10000 ** (2 * (i//2) / d_model) for i in range(d_model)]
                                   for pos in range(max_seq_len)], requires_grad=False)
        pos_encode[:, 0::2] = torch.sin(pos_encode[:, 0::2])
        pos_encode[:, 1::2] = torch.cos(pos_encode[:, 1::2])
        # - pos_encode: (seq_len, d_model)
        return pos_encode

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, p_drop) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)
        self.sublayer1_prenorm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.sublayer2_prenorm = nn.LayerNorm(d_model)
        self.pos_wise_ffn = FeedForwardNetwork(d_model)

    def forward(self, x, src_mask):
        res, x_ln = x, self.sublayer1_prenorm(x)
        x = res + self.dropout(self.self_attn(
            q=x_ln, k=x_ln, v=x_ln,
            mask=src_mask.unsqueeze(1).unsqueeze(1)))
        res, x_ln = x, self.sublayer2_prenorm(x)
        x = res + self.dropout(self.pos_wise_ffn(x_ln))
        return x

class MultiHeadAttention(nn.Module):
    # - src_embed_dim = d_model
    def __init__(self, d_model, n_head) -> None:
        super().__init__()
        self.n_head, self.one_head_dim = n_head, d_model // n_head
        self.w_q = nn.Linear(d_model, self.one_head_dim * self.n_head, bias=True)
        self.w_k = nn.Linear(d_model, self.one_head_dim * self.n_head, bias=True)
        self.w_v = nn.Linear(d_model, self.one_head_dim * self.n_head, bias=True)
        self.w_out = nn.Linear(self.one_head_dim * self.n_head, d_model, bias=True)

    def forward(self, q, k, v, mask=None):
        # - x: (batch_size, seq_len, d_model)
        batch_size, q_len, kv_len = q.size(0), q.size(1), k.size(1)
        Q = self.w_q(q).view(batch_size, q_len, self.n_head, 
                             self.one_head_dim).transpose(1, 2)
        K = self.w_k(k).view(batch_size, kv_len, self.n_head,
                             self.one_head_dim).transpose(1, 2)
        V = self.w_v(v).view(batch_size, kv_len, self.n_head,
                             self.one_head_dim).transpose(1, 2)
        # - Q, K, V: (batch_size, n_head, seq_len, one_head_dim)

        Q_KT = torch.matmul(Q, torch.transpose(K, 2, 3))

        if mask != None:
            Q_KT.masked_fill_(mask, -1e9)

        attn = F.softmax(Q_KT / self.one_head_dim ** 0.5, dim=-1)

        O = self.w_out(torch.matmul(attn, V).transpose(1, 2).reshape(
                batch_size, q_len, self.one_head_dim * self.n_head))
        # - O: (batch_size, seq_len, d_model)
        return O


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model, bias=True)
        self.linear2 = nn.Linear(4 * d_model, d_model, bias=True)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerEncoder(nn.Module):
    def __init__(self, n_labels, padding_idx, n_head, d_model, 
                 n_layers, p_drop, max_seq_len=10) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model
        self.dropout = nn.Dropout(p=p_drop)
        self.input_embedding = nn.Embedding(
            num_embeddings=n_labels + 1, embedding_dim=d_model, padding_idx=padding_idx)
        self.positional_encode = PositionalEncode(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, p_drop) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model) # for memory
        self.out_layer = nn.Linear(d_model, n_labels, bias=True)


    def forward(self, src_tokens, **kwargs):
        src_mask = src_tokens.eq(self.padding_idx)
        src_lens = src_tokens.ne(self.padding_idx).long().sum(dim=-1)
        # - src_embed: (batch_size, src_len, d_model)
        src_embed = self.input_embedding(src_tokens) * (self.d_model ** 0.5)
        x = self.dropout(self.positional_encode(src_embed))
        for layer in self.layers:
            x = layer(x, src_mask)
        encoder_out = self.layer_norm(x)
        final_out = torch.zeros((encoder_out.shape[0], 1, encoder_out.shape[2]), device=encoder_out.device)
        for b in range(len(encoder_out)):
            final_out[b, 0, :] = encoder_out[b, src_lens[b]-1, :]
        final_out = self.out_layer(final_out)
        return final_out.squeeze()

str2rnn = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    
class LanguageModel(nn.Module):
    def __init__(self, n_labels, n_layers, d_model, p_drop, padding_idx, rnn_type):
        super().__init__()
        self.padding_idx = padding_idx
        self.embed = nn.Embedding(n_labels + 1, d_model, padding_idx=padding_idx)
        self.rnn = str2rnn[rnn_type](input_size=d_model, hidden_size=d_model, num_layers=n_layers, dropout=p_drop, batch_first=True)
        self.out_layer = nn.Linear(d_model, n_labels, bias=True)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x):
        src_lens = x.ne(self.padding_idx).long().sum(dim=-1)
        src_embed = self.dropout(self.embed(x))
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_encoder_out, _ = self.rnn(packed_src_embed)
        # - encoder_out: (batch_size, src_len, d_model) where 3rd is last layer [h_fwd; (h_bkwd)]
        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(packed_encoder_out, batch_first=True)
        final_out = torch.zeros((encoder_out.shape[0], 1, encoder_out.shape[2]), device=encoder_out.device)
        for b in range(len(encoder_out)):
            final_out[b, 0, :] = encoder_out[b, src_lens[b]-1, :]
        final_out = self.out_layer(final_out)
        return final_out.squeeze()

def process_lm_data(file_path, with_label=True, shuf=True):
    dataset = list()
    csv_reader = csv.reader(open(file_path))
    for row in csv_reader:
        if len(str(row[2])) != 0:
            if with_label:
                x = str(row[2])[:-1]
                y = str(row[2])[-1]
            else:
                x = str(row[2])

        if with_label:
            dataset.append([x, y])
        else:
            dataset.append([x])

    if with_label:
        data = pd.DataFrame(dataset, columns = ['x', 'y'])[1:]
    else:
        data = pd.DataFrame(dataset, columns = ['x'])[1:]

    if shuf:
        data = shuffle(data)

    return data

class LanguageModelDataset(Dataset):
    def __init__(self, x, y, padding_idx=-1):
        self.x = x
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.padding_idx = padding_idx
        self.max_len = max(list(len(x) for x in self.x))
  
    def __getitem__(self, index):
        item = dict()
        x = [int(s) -1  for s in self.x[index]]
        if len(x) < self.max_len:
            x.extend([self.padding_idx] * (self.max_len - len(x)))
        item['x'] = torch.tensor(x)

        if self.y is not None:
            item['y'] = torch.tensor(int(self.y[index]) - 1)
        return item

    def __len__(self):
        return len(self.x)


def cal_performance(preds, labels):
    report = classification_report(labels, preds, zero_division=0, output_dict=True)
    acc = report['accuracy']
    f_score = report['macro avg']['f1-score']
    return acc, f_score

def train_epoch(model, criterion, optim, scheduler, train_loader, val_loader, epoch, train_log_interval=10, val_internal=50, val_res=None, save_dir=None, model_name=None, device=0):
    model.train()
    len_iter = len(train_loader)
    n_step = 0
    val_acces, val_fscores, val_losses = [], [], []
    for i, batch in enumerate(train_loader, start=1):
        optim.zero_grad()
        x, y = batch['x'].to(device), batch['y'].to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optim.step()

        n_step += 1
        if scheduler:
            scheduler.step()
        if i % train_log_interval == 0:
            print("epoch: %d [%d/%d], loss: %.6f, lr: %.8f, steps: %d" %
                  (epoch, i, len_iter, loss.item(), optim.param_groups[0]["lr"], n_step + len_iter * (epoch-1)))
        if i % val_internal == 0:
            acc, f_score, loss = val_epoch(model, criterion, val_loader, save_dir, model_name, val_res, device)
            val_acces.append(acc)
            val_fscores.append(f_score)
            val_losses.append(loss)

    return val_acces, val_fscores, val_losses

def val_epoch(model, criterion, val_loader, save_dir, model_name, val_res, device):
    model.eval()
    total_eval_loss = 0
    preds, Labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_eval_loss += loss.item()
            batch_preds = torch.argmax(pred, dim=-1).detach().cpu().tolist()
            label_ids = y.to('cpu').numpy().tolist()
            preds.extend(batch_preds)
            Labels.extend(label_ids)

    avg_val_loss =total_eval_loss / len(val_loader)
    acc, f_score = cal_performance(preds, Labels)
    if save_dir:
        if f_score > max(val_res):
            save_model(model, save_dir, model_name)
    val_res.append(f_score)
    print("Valid | acc: %.4f, f_score: %.4f, global optim: %.4f, loss: %.4f" % (acc, f_score, max(val_res), avg_val_loss))
    return acc, f_score, avg_val_loss

def save_model(model, save_dir, model_name):
    output_model_file = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), output_model_file)
    print('Model has save to %s' % output_model_file)

def train(model, criterion, optim, scheduler, train_loader, val_loader, n_epoch, save_dir, model_name, device):
    val_res = [0]
    for i in range(1, n_epoch + 1):
        train_epoch(model, criterion, optim, scheduler, train_loader, val_loader, save_dir=save_dir, model_name=model_name, epoch=i, train_log_interval=50,  val_internal=100, val_res=val_res, device=device)
        val_epoch(model, criterion, val_loader, save_dir, model_name, val_res, device)

def get_lm_preds(alpha=0.5, device='cpu'):
    
    model_paths = ['./gru.bin', './transformer.bin']#, './lstm.bin']
    gru = LanguageModel(6, n_layers=1, d_model=8, p_drop=0.05, padding_idx=6, rnn_type='gru')
    gru.load_state_dict(torch.load(model_paths[0], map_location=device))
    
    if len(model_paths) > 1:
        transformer_enc = TransformerEncoder(n_labels=6, padding_idx=6, n_head=4, d_model=16, n_layers=2, p_drop=0.2, max_seq_len=10)
        transformer_enc.load_state_dict(torch.load(model_paths[1], map_location=device))
    
    if len(model_paths) > 2:
        lstm = LanguageModel(6, n_layers=1, d_model=16, p_drop=0, padding_idx=6, rnn_type='lstm')
        lstm.load_state_dict(torch.load(model_paths[2], map_location=device))
    
    batch_size = 32
    data = process_lm_data('./test_data_new.csv', with_label=False, shuf=False)
    
    test_dataset = LanguageModelDataset(data['x'].tolist(), None, padding_idx=6)
    
    def test_epoch(model, test_loader, device):
        model.to(device)
        model.eval()
        preds = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader, start=1):
                x = batch['x'].to(device)
                logits = model(x)
                batch_preds = F.softmax(logits, dim=-1) * alpha
                preds.append(batch_preds)
        return torch.cat(preds)
    models_preds = []
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    models_preds.append(test_epoch(gru, test_loader, device))
    
    if len(model_paths) > 1:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        models_preds.append(test_epoch(transformer_enc, test_loader, device))
    
    if len(model_paths) > 2:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        models_preds.append(test_epoch(lstm, test_loader, device))
        
    avg_preds = sum(models_preds) / len(model_paths)
    
    return avg_preds
        
if __name__ == '__main__':  
    data = process_lm_data('./train_data.csv', with_label=True, shuf=True)
    X_train, X_val, y_train, y_val = train_test_split(data['x'].tolist(), data['y'].tolist(), test_size=0.05, random_state=1)

    batch_size = 32
    train_dataset = LanguageModelDataset(X_train, y_train, padding_idx=6)
    val_dataset = LanguageModelDataset(X_val, y_val, padding_idx=6)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cpu')
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    #model = LanguageModel(6, n_layers=1, d_model=8, p_drop=0.05, padding_idx=6, rnn_type='gru')
    #model = TransformerEncoder(n_labels=6, padding_idx=6, n_head=4, d_model=16, n_layers=2, p_drop=0.2, max_seq_len=10)
    model = LanguageModel(6, n_layers=1, d_model=16, p_drop=0, padding_idx=6, rnn_type='lstm')

    #optim = torch.optim.SGD(model.parameters(), lr=1e-2) # SGD: 2e-3, 5e-3
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3)
    model_name = 'lstm.bin'
    train(model, criterion, optim, None, train_loader, val_loader, n_epoch=200, save_dir='.', model_name=model_name, device=device)