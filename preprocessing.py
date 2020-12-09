import pandas as pd
import csv
from datetime import datetime
import nltk

# data_subset = 'train'
data_subset = 'test'
dateparse = lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')

cols = ['tweet id', 'user handle', 'tweet text', 'time stamp', 'device']
if data_subset == 'test':
    cols = cols[1:-1]  # there's no label
dataset = pd.read_csv('trump_%s.tsv' % data_subset, sep='\t', names=cols)
dataset.dropna(inplace=True)
if data_subset == 'train':
    dataset.drop(columns=['tweet id'], inplace=True)
dataset['time stamp'] = dataset['time stamp'].apply(dateparse)
if data_subset == 'train':
    label_col = (dataset['device'] == 'android') & (dataset['time stamp'] < datetime(2017, 4, 1))
    dataset['label'] = label_col.astype(int)
dataset.to_csv('dataset_%s.csv' % data_subset, index=False)


def tokenize(text):
    valid_tokens = []
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        token = token.lower()
        if token.replace('.', '').isalpha():
            valid_tokens.append(token)
    return valid_tokens
