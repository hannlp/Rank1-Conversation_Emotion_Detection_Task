import os
import torch
import torch.nn as nn
from main import set_seed, process_train, load_model_and_tokenizer, split_and_load_dataset, ChildTuningAdamW, get_linear_schedule_with_warmup, train

if __name__ == '__main__':
    n_epoch = 5
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    set_seed(1)

    data, n_labels, cnt = process_train('./train_data.csv')
    #data, n_labels = get_aug_data('./eda_to_aug.csv')
    #Counter({6: 17549, 2: 6515, 1: 5651, 5: 4138, 3: 2534, 4: 432})
    #data = shuffle(data)
    criterion = nn.CrossEntropyLoss()

    model, tokenizer = load_model_and_tokenizer('./checkpoints', device, n_labels=n_labels)

    *_, train_loader, val_loader = split_and_load_dataset(data, tokenizer, max_len=64, batch_size=32, test_size=0.05)

    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=len(train_loader) * n_epoch)

    save_dir = './checkpoints_sgdm'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train(model, criterion, optim, scheduler, train_loader, val_loader, n_epoch=n_epoch, save_dir=save_dir, device=device)