import os
import numpy as np
import random
import torch
import torch.nn as nn
from data_aug import get_aug_data
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification,AutoTokenizer, get_linear_schedule_with_warmup
from data import process_train, make_batch, split_and_load_dataset, shuffle
from utils import get_logger
from optims import ChildTuningAdamW
from sklearn.metrics import classification_report

logger = get_logger('./logs', __name__)

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer(model_path, device, p_dropout=0.1, n_labels=None):
    logger.info("loading model and tokenizer from {} ...".format(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(model_path)
    # if n_labels:
    #     config.num_labels = n_labels
    if hasattr(config, 'hidden_dropout_prob'):
        config.hidden_dropout_prob = p_dropout
    if hasattr(config, 'classifier_dropout'):
        config.classifier_dropout = p_dropout
    if hasattr(config, 'attention_probs_dropout_prob'):
        config.attention_probs_dropout_prob = p_dropout

    # model = AutoModelForSequenceClassification.from_config(config)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

    #freeze paras
    # for name, para in model.named_parameters():
    #     if name.find('embedding') != -1:
    #         para.requires_grad = False

    # for para in model.parameters():
    #     para.requires_grad = False

    # for name, para in model.named_parameters():
    #     if name.find('layer') != -1:
    #         para.requires_grad = False

    if n_labels:
        hidden_size = model.config.hidden_size
        if hasattr(model.classifier, 'out_proj'):
            model.classifier.out_proj = nn.Linear(in_features=hidden_size, out_features=n_labels, bias=True)
        else:
            model.classifier = nn.Linear(in_features=hidden_size, out_features=n_labels, bias=True)
        #model.classifier = Highway(in_features=hidden_size, out_features=n_labels, func=F.relu, bias=True)
        model.config.num_labels = n_labels
    model.to(device)
    return model, tokenizer

def cal_performance(preds, labels, score_type='recall'):
    # print(preds, labels)
    # pred_flat = np.argmax(np.array(preds), axis=-1).flatten()
    # labels_flat = np.array(labels).flatten()
    #
    # print(pred_flat, labels_flat)
    report = classification_report(labels, preds, zero_division=0, output_dict=True)
    acc = report['accuracy']
    #score = report['weighted avg']['f1-score']
    if score_type == 'macro-f1':
        score = report['macro avg']['f1-score']
    else:
        score = report['macro avg']['recall']
    return acc, score

def train_epoch(model, criterion, optim, scheduler, train_loader, val_loader, epoch, train_log_interval=10, val_internal=50, val_res=None, save_dir=None, device=0):
    model.train()
    len_iter = len(train_loader)
    n_step = 0
    val_acces, val_fscores, val_losses = [], [], []
    for i, batch in enumerate(train_loader, start=1):
        optim.zero_grad()
        input_ids, attention_mask, labels = make_batch(batch, device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optim.step()

        n_step += 1
        if scheduler:
            scheduler.step()
        if i % train_log_interval == 0:
            logger.info("epoch: %d [%d/%d], loss: %.6f, lr: %.8f, steps: %d" %
                  (epoch, i, len_iter, loss.item(), optim.param_groups[0]["lr"], n_step + len_iter * (epoch-1)))
        if i % val_internal == 0:
            acc, score, loss = val_epoch(model, criterion, val_loader, save_dir, val_res, device)
            val_acces.append(acc)
            val_fscores.append(score)
            val_losses.append(loss)

    return val_acces, val_fscores, val_losses

def val_epoch(model, criterion, val_loader, save_dir, val_res, device):
    model.eval()
    total_eval_loss = 0
    preds, Labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = make_batch(batch, device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_eval_loss += loss.item()
            batch_preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist()
            label_ids = labels.to('cpu').numpy().tolist()
            preds.extend(batch_preds)
            Labels.extend(label_ids)

    avg_val_loss =total_eval_loss / len(val_loader)
    acc, score = cal_performance(preds, Labels)
    if save_dir:
        if score > max(val_res) and acc > 0.76 :
            save_model(model, save_dir)
    val_res.append(score)
    logger.info("Valid | acc: %.4f, score: %.4f, global optim: %.4f, loss: %.4f" % (acc, score, max(val_res), avg_val_loss))
    return acc, score, avg_val_loss

def save_model(model, save_dir):
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_dir, CONFIG_NAME)
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_dir)
    logger.info('Model has save to %s' % save_dir)

def train(model, criterion, optim, scheduler, train_loader, val_loader, n_epoch, save_dir, device):
    val_res = [0]
    for i in range(1, n_epoch + 1):
        train_epoch(model, criterion, optim, scheduler, train_loader, val_loader, save_dir=save_dir, epoch=i, train_log_interval=10, val_internal=20, val_res=val_res, device=device)
        val_epoch(model, criterion, val_loader, save_dir, val_res, device)

if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    data, n_labels, cnt = process_train('./train_data.csv') #Counter({6: 17549, 2: 6515, 1: 5651, 5: 4138, 3: 2534, 4: 432})
    #data, n_labels = get_aug_data('./eda_to_aug.csv')
    criterion = nn.CrossEntropyLoss()

    seeds = [222]#[111]#, 222, 333, 444, 555]#, 666, 777, 888, 999, 1111]#, 2222, 3333, 4444, 5555]#
    for seed in seeds:
        set_seed(seed)
        n_epoch = 3

        model_name = 'bert'
        if model_name == 'bert':
            model, tokenizer = load_model_and_tokenizer('./bert-base-chinese', device, p_dropout=0.1, n_labels=n_labels) #p_drop 0.1
        elif model_name == 'roberta':
            model, tokenizer = load_model_and_tokenizer('./roberta-base-finetuned-dianping-chinese', device,
                                                        p_dropout=0.05,
                                                        n_labels=n_labels)
        elif model_name == 'electra':
            model, tokenizer = load_model_and_tokenizer('./chinese-electra-180g-large-discriminator', device,
                                                        p_dropout=0.3,
                                                        n_labels=n_labels)
        elif model_name == 'roberta-jd':
            model, tokenizer = load_model_and_tokenizer('./roberta-base-finetuned-jd-full-chinese', device,
                                                        p_dropout=0.05,
                                                        n_labels=n_labels)
            
        save_dir = './checkpoints/{}/{}'.format(model_name, seed)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        *_, train_loader, val_loader = split_and_load_dataset(data, tokenizer, max_len=84, batch_size=32, test_size=0.1) # 0.05
        optim = torch.optim.AdamW(model.parameters(), lr=2e-5)  # 5e-6
        # optim = ChildTuningAdamW(model.parameters(), lr=2e-5, mode='ChildTuning-D', reserve_p=0.1,
        #                          model=model, criterion=criterion, train_loader=train_loader, device=device)
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=400, num_training_steps=len(train_loader) * n_epoch)
        train(model, criterion, optim, scheduler, train_loader, val_loader, n_epoch=n_epoch, save_dir=save_dir, device=device)

        del model, tokenizer, optim, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        data = shuffle(data)