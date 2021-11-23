import pandas as pd
from data import process_train

def prepare_to_aug(in_path, out_path):
    data, *_ = process_train(in_path, shuf=False)
    data.to_csv(out_path, sep='\t', columns=['Labels', 'Text'], index=False, header=False)

#nohup python data_aug/code/augment.py --input to_aug.csv --num_aug 3 &

def get_aug_data(path):
    data = pd.read_csv(path, sep='\t', names=['Labels', 'Text'])
    data['Labels'] = data['Labels'].astype(int)
    n_labels = len(set(data.Labels))
    #data.sample(frac=1)
    return data, n_labels

if __name__ == '__main__':
    prepare_to_aug(in_path='./train_data.csv', out_path='./to_aug.csv')

