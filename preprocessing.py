import pandas as pd
import csv
from datetime import datetime


dateparse = lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')

cols = ['tweet id', 'user handle', 'tweet text', 'time stamp', 'device']
dataset = pd.read_csv('trump_train.tsv', sep='\t', names=cols)
dataset.dropna(inplace=True)
dataset.drop(columns=['tweet id'], inplace=True)
dataset['time stamp'] = dataset['time stamp'].apply(dateparse)
dataset['label'] = (dataset['device'] == 'android') & (dataset['time stamp'] < datetime(2017, 4, 1))

print('done')
