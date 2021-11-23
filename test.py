import torch
import torch.nn as nn
from main import cal_performance, load_model_and_tokenizer
from sklearn.metrics import classification_report, confusion_matrix
from data import split_and_load_dataset, process_test, make_batch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from time_series import get_lm_preds

def test(model, tokenizer, idx2label, criterion, test_loader, device, with_label=True):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    result = []
    preds = []
    Labels = [] if with_label else None
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = make_batch(batch, device)
            outputs = model(input_ids, attention_mask=attention_mask)
            if with_label:
                loss = criterion(outputs.logits, labels)
            logits = outputs.logits

            pred = torch.argmax(logits, dim=-1).tolist()
            for i in range(len(input_ids)):
                #text = ''.join(tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True))
                cls = idx2label[pred[i]]
                #result.append([text, cls])
                preds.append(cls)
                if with_label:
                    Labels.append(idx2label[labels[i].item()])


            logits = logits.detach().cpu().numpy()
            # if with_label:
            #     label_ids = labels.to('cpu').numpy()
            #     total_eval_loss += loss.item()
            #     total_eval_accuracy += cal_performance(logits, label_ids)

    if with_label:
        avg_val_accuracy = total_eval_accuracy / len(test_loader)
        print("Accuracy: %.4f" % (avg_val_accuracy))
        print("Average test loss: %.4f" % (total_eval_loss / len(test_loader)))
        print("-------------------------------")
    return result, preds, Labels

def gen_res(preds):
    index = list(range(1, len(preds) + 1))
    d = {'ID': index, 'Last Label': preds}
    df = pd.DataFrame(d)
    df.to_csv('./res.csv', sep=',', columns=['ID', 'Last Label'], index=False, header=True)
    print('has save to {}'.format('./res.csv'))


def get_preds_and_labels(model, test_loader, device, with_label=True):
    model.eval()
    preds = []
    Labels = [] if with_label else None
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = make_batch(batch, device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = F.softmax(logits, dim=-1)
            preds.append(batch_preds)
            if with_label:
                Labels.extend(labels.tolist())
    total_preds = torch.cat(preds)
    return total_preds, Labels

if __name__ == '__main__':
    with_label = False
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # 使用自己划分的测试集
    #data, n_labels, cnt, idx2label = process_test('./train_data.csv')

    # 使用标准测试集
    _, n_labels, cnt, idx2label = process_test('./train_data.csv')
    data = process_test('./test_data_new.csv', with_label=with_label)
    criterion = nn.CrossEntropyLoss()

    lm_weight = 2.8#2.7
    
    single_model = False
    if single_model:
        model_path = './checkpoints/electra/111' #'./checkpoints'
        model, tokenizer = load_model_and_tokenizer(model_path, device)
        *_,  test_loader = split_and_load_dataset(data, tokenizer, max_len=64, batch_size=32, with_label=with_label, test_size=1.0, shuf=False)
        result, preds, Labels = test(model, tokenizer, idx2label, criterion, test_loader, device=device, with_label=with_label)
        if with_label:
            print(classification_report(preds, Labels))
            print(confusion_matrix(Labels, preds))
        gen_res(preds)
    else:
        # model_names = ['bert', 'roberta', 'electra']#, 'roberta-jd']
        # model_paths = ['./checkpoints/{}/555'.format(model_name) for model_name in model_names]
        
        # model_paths = ['./checkpoints/bert/555', './checkpoints/roberta/555', './checkpoints/electra/222', './checkpoints/bert/111', './checkpoints/roberta/111', './checkpoints/electra/111']
        model_paths = ['./checkpoints/bert/555', './checkpoints/roberta/555', './checkpoints/electra/222']
        #model_weights = [0.3, 0.5, 0.2]

        # model_names = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1111]#, 2222, 3333, 4444, 5555]#, 999, 1111] 2:0.7289 3:0.7311 4:0.7327 5:0.7336 6:0.7345 7:0.7329 8:0.7350 14:0.7352
        # model_paths = ['./checkpoints/electra/{}'.format(model_name) for model_name in model_names]

        # model_names = ['bert', 'roberta', 'electra']
        # seeds = [111, 222]#, 333, 444, 555, 666, 777, 888, 999, 1111]
        # model_paths = ['./checkpoints/{}/{}'.format(model_name, seed) for seed in seeds for model_name in model_names]

        n_models = len(model_paths)
        
        models_preds = []
        for model_path in tqdm(model_paths):

            model, tokenizer = load_model_and_tokenizer(model_path, device)
            *_, test_loader = split_and_load_dataset(data, tokenizer, max_len=64, batch_size=32, with_label=with_label, test_size=1.0, shuf=False)
            single_model_preds, Labels = get_preds_and_labels(model, test_loader, device, with_label=with_label)
            models_preds.append(single_model_preds)
            del model, tokenizer, test_loader
            
        if lm_weight != 0:
            print('combine with language model.')
            lm_preds = get_lm_preds(alpha=lm_weight, device=device)
            models_preds.append(lm_preds)
            n_models += lm_weight
        
        avg_preds = sum(models_preds) / n_models
        #avg_preds = sum([p*w for p, w in zip(models_preds, model_weights)])
        #print(list(avg_preds))
        preds = torch.argmax(avg_preds, dim=-1).tolist()
        # if with_label:
        #     print(cal_performance(avg_preds.cpu().numpy(), np.array(Labels)))

        real_preds = list(map(lambda x: idx2label[x], preds))
        gen_res(real_preds)