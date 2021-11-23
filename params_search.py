import torch
import torch.nn as nn
from utils import get_logger
from main import (
    process_train, set_seed, load_model_and_tokenizer,
    split_and_load_dataset, get_linear_schedule_with_warmup,
    train_epoch, ChildTuningAdamW
    )
import argparse

logger = get_logger('./logs', __name__)

def hyper_params_search(model_path, data_path, num_warmup_steps=400, p_dropout=0.1, lr=2e-5, max_len=64, batch_size=32, optim_type='adamw', reserve_p=None, device=2):
    n_epoch = 3
    device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
    data, n_labels, _ = process_train(data_path)

    def evaluate(res: list, high_is_best=True):
        sorted_res = sorted(res, reverse=high_is_best)
        topk = [1]
        topk_avg_list = [sum(sorted_res[:k]) / k for k in topk]
        return topk_avg_list

    seeds = [111, 222, 333, 444, 555]
    acc_res, loss_res = [], []
    for seed in seeds:
        set_seed(seed)
        acc, loss = [], []
        model, tokenizer = load_model_and_tokenizer(model_path, device, p_dropout, n_labels=n_labels)
        *_, train_loader, val_loader = split_and_load_dataset(data, tokenizer, max_len=max_len, batch_size=batch_size, test_size=0.05)
        criterion = nn.CrossEntropyLoss()
        if optim_type == 'adamw':
            optim = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optim_type == 'child-tuning-adamw-f':
            optim = ChildTuningAdamW(model.parameters(), lr=lr, mode='ChildTuning-F', reserve_p=reserve_p,
                                     model=model, criterion=criterion, train_loader=train_loader, device=device)
        elif optim_type == 'child-tuning-adamw-d':
            optim = ChildTuningAdamW(model.parameters(), lr=lr, mode='ChildTuning-D', reserve_p=reserve_p,
                                     model=model, criterion=criterion, train_loader=train_loader, device=device)

        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_loader) * n_epoch)
        val_acc = [0]
        for i in range(1, n_epoch + 1):
            val_acces, val_losses = train_epoch(model, criterion, optim, scheduler, train_loader, val_loader,
                                                epoch=i, train_log_interval=10, val_internal=20, val_acc=val_acc, device=device)
            acc.extend(val_acces)
            loss.extend(val_losses)

        tok_acc_avg_list = evaluate(acc)
        tok_loss_avg_list = evaluate(loss, high_is_best=False)
        acc_res.append(tok_acc_avg_list)
        loss_res.append(tok_loss_avg_list)
        del optim, model, tokenizer, train_loader, val_loader, criterion, scheduler
        torch.cuda.empty_cache()

    topk_avg_acc = [round(sum(t) / len(seeds), 3) for t in zip(*acc_res)]
    topk_avg_loss = [round(sum(t) / len(seeds), 3) for t in zip(*loss_res)]

    result = {'model': model_path, 'warmup_steps': num_warmup_steps, 'p_drop': p_dropout,
              'lr':lr, 'max_len': max_len, 'batch_size': batch_size, 'optim_type': optim_type,
              'reserve_p': reserve_p, 'topk_avg_acc': topk_avg_acc, 'topk_avg_loss': topk_avg_loss}

    logger.info(result)
    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-group", type=int, default=1)
    parser.add_argument("-device", type=int, default=0)
    args = parser.parse_args()

    hyper_params_space = {'model':['./roberta-base-finetuned-dianping-chinese', './bert-base-chinese', './chinese-electra-180g-large-discriminator'],
                          'warmup_steps': [0, 100, 200, 400, 600, 800],
                          'p_dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                          'lr': [5e-6, 1e-5, 2e-5, 4e-5, 6e-5],
                          'reserve_p': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]}

    if args.group == 1:
        for model_path in hyper_params_space['model']:
            for p_drop in hyper_params_space['p_dropout']:
                hyper_params_search(model_path=model_path, data_path='./train_data.csv',
                                    num_warmup_steps=400, p_dropout=p_drop, lr=2e-5, max_len=64, batch_size=32, device=args.device)
    elif args.group == 2:
        for model_path in hyper_params_space['model']:
            for lr in hyper_params_space['lr']:
                hyper_params_search(model_path=model_path, data_path='./train_data.csv',
                                    num_warmup_steps=400, p_dropout=0.1, lr=lr, max_len=64, batch_size=32, device=args.device)
    elif args.group == 3:
        for model_path in hyper_params_space['model']:
            for lr in hyper_params_space['lr']:
                for warmup_steps in hyper_params_space['warmup_steps']:
                    hyper_params_search(model_path=model_path, data_path='./train_data.csv',
                                        num_warmup_steps=warmup_steps, p_dropout=0.1, lr=lr, max_len=64, batch_size=32, device=args.device)
    elif args.group == 4:
        for lr in hyper_params_space['lr']:
            for warmup_steps in hyper_params_space['warmup_steps']:
                hyper_params_search(model_path='./chinese-electra-180g-large-discriminator', data_path='./train_data.csv',
                                    num_warmup_steps=warmup_steps, p_dropout=0.3, lr=lr, max_len=64, batch_size=32,
                                    device=args.device)
    elif args.group == 5:
        for lr in hyper_params_space['lr']:
            for warmup_steps in hyper_params_space['warmup_steps']:
                hyper_params_search(model_path='./roberta-base-finetuned-dianping-chinese', data_path='./train_data.csv',
                                    num_warmup_steps=warmup_steps, p_dropout=0.05, lr=lr, max_len=64, batch_size=32,
                                    device=args.device)
    elif args.group == 6:
        for lr in hyper_params_space['lr']:
            for warmup_steps in hyper_params_space['warmup_steps']:
                hyper_params_search(model_path='./bert-base-chinese', data_path='./train_data.csv',
                                    num_warmup_steps=warmup_steps, p_dropout=0.1, lr=lr, max_len=64, batch_size=32,
                                    device=args.device)
    elif args.group == 7:
        for model_path in hyper_params_space['model']:
            for reserve_p in hyper_params_space['reserve_p']:
                hyper_params_search(model_path=model_path, data_path='./train_data.csv',
                                    num_warmup_steps=400, p_dropout=0.1, lr=2e-5, max_len=64, batch_size=32,
                                    optim_type='child-tuning-adamw-f', reserve_p=reserve_p, device=args.device)
                hyper_params_search(model_path=model_path, data_path='./train_data.csv',
                                    num_warmup_steps=400, p_dropout=0.1, lr=2e-5, max_len=64, batch_size=32,
                                    optim_type='child-tuning-adamw-d', reserve_p=reserve_p, device=args.device)