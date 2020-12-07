import re
import string

import pandas as pd
import csv
from datetime import datetime

from nltk.corpus import stopwords

dateparse = lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')

cols = ['tweet id', 'user handle', 'tweet text', 'time stamp', 'device']
dataset = pd.read_csv('trump_train.tsv', sep='\t', names=cols)
dataset.dropna(inplace=True)
dataset.drop(columns=['tweet id'], inplace=True)
dataset['time stamp'] = dataset['time stamp'].apply(dateparse)
dataset['label'] = (dataset['device'] == 'android') & (dataset['time stamp'] < datetime(2017, 4, 1))

print('done')

# Globals
stop_words = set(stopwords.words('english'))


def create_tweets_df(tweets_src_file):
    dt_tweets = pd.read_csv(tweets_src_file, sep='\t',
                            names=['tweet id', 'user handle', 'tweet text', 'time stamp', 'device'])
    dt_tweets.dropna(inplace=True)
    dt_tweets.drop(columns=['tweet id'], inplace=True)
    dt_tweets['time stamp'] = dt_tweets['time stamp'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    dt_tweets['label'] = (dt_tweets['device'] == 'android') & (dt_tweets['time stamp'] < datetime(2017, 4, 1))
    return dt_tweets


def get_raw_data():
    dt_tweets_df = create_tweets_df('trump_train.tsv')
    dt_tweets_df['tweet text preprocessed'] = dt_tweets_df['tweet text'].apply(normalize_text)
    dt_tweets_df = dt_tweets_df.sort_values(['time stamp']).reset_index()
    label = dt_tweets_df['label']
    dt_tweets_df = dt_tweets_df.drop(['label'], axis=1)
    return dt_tweets_df, label


def normalize_text(text):
    """
    This function takes as input a text on which several
    NLTK algorithms will be applied in order to preprocess it
    """
    pattern = r'''(?x)          # set flag to allow verbose regexps
           (?:[A-Z]\.)+          # abbreviations, e.g. U.S.A.
           | \w+(?:-\w+)*        # words with optional internal hyphens
           | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
           | \.\.\.              # ellipsis
           | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
           '''
    text = text.lower().translate(string.punctuation)
    regexp = re.compile(pattern)
    tokens = regexp.findall(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
