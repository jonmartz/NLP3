import re
import string
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime
from nltk.corpus import stopwords
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics

dateparse = lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')

# Globals
stop_words = set(stopwords.words('english'))
tweet_id, handle, tweet_text, tweet_time, device, processed_tweet_text = 'tweet id', 'user handle', 'tweet text', \
                                                                         'time stamp', 'device', 'tweet text processed'


def create_tweets_df(tweets_src_file):
    """
    Create pandas DF from the tsv file, with column names and label column (if train).
    :param tweets_src_file: path of tsv file
    :return: pd.Dataframe
    """
    dt_tweets = pd.read_csv(tweets_src_file, sep='\t')
    is_train_set = len(dt_tweets.columns) == 5
    if is_train_set:
        dt_tweets.columns = ['tweet id', 'user handle', 'tweet text', 'time stamp', 'device']
        dt_tweets.drop(columns=['tweet id'], inplace=True)
    else:  # this is a test set
        dt_tweets.columns = ['user handle', 'tweet text', 'time stamp']
    dt_tweets.dropna(inplace=True)
    dt_tweets['time stamp'] = dt_tweets['time stamp'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    if is_train_set:
        dt_tweets['label'] = (dt_tweets['device'] == 'android') & (dt_tweets['time stamp'] < datetime(2017, 4, 1))
    return dt_tweets


def get_raw_data(path):
    """
    Add processed text to the df and get label column.
    :param path: of tsv file
    :return: data (pd.Dataframe) and labels (pd.Series)
    """
    df = create_tweets_df(path)
    df['tweet text processed'] = df['tweet text'].apply(normalize_text)
    df = df.sort_values(['time stamp']).reset_index()
    if 'label' in df.columns:  # for train set
        label = df['label']
        df = df.drop(['label'], axis=1)
    else:  # for test set
        label = None
    return df, label


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


def remove_web_links(token):
    token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                   '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
    return token


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        if item not in stop_words:
            stems.append(item)
    return stems


def tokenize_for_vectors(text):
    punct = ['!', '?', '@', '#', '$', '%', '&', '(', ')', '-', '=', '+', '/', ':', ';', "'", '"', ',', '.']
    tokens = nltk.word_tokenize(text.lower())
    selected_tokens = []
    for token in tokens:
        # accept only alphabetic words (with hyphens too), abbreviations and simple punctuations
        if token in punct or token.replace('.', '').replace('-', '').isalpha():
            selected_tokens.append(token)
    return selected_tokens


class ItemSelector(BaseEstimator, TransformerMixin):
    """
    Returns the desired column from the dataframe
    """
    def __init__(self, key):
        """
        :param key: name of column to extract
        """
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TimeStatistics:
    """
    Extract features related to time.
    """
    def __init__(self, key):
        self.features_name = ['weekday', 'hour']
        self.key = key

    def get_feature_names(self, ):
        return self.features_name

    @staticmethod
    def get_relative_time_to_election(time_list):
        election_date = datetime.strptime("8-11-2016", '%d-%m-%Y')
        relative_to_election_date = list(map(lambda x: 1 if x > election_date else 0, time_list))
        return np.array(relative_to_election_date).reshape(-1, 1)

    @staticmethod
    def get_weekday(time_list):
        week_days_for_time_list = [time.weekday() for time in time_list]
        week_days_for_time_list = [week_day / 7 for week_day in week_days_for_time_list]
        return np.array(week_days_for_time_list).reshape(-1, 1)

    @staticmethod
    def get_year(time_list):
        year_for_time_list = [time.year for time in time_list]
        years = list(set(year_for_time_list))
        year_for_time_list = [years.index(year) / len(years) for year in year_for_time_list]
        return np.array(year_for_time_list).reshape(-1, 1)

    @staticmethod
    def get_hour(time_list):
        hour_for_time_list = [time.hour for time in time_list]
        hours = list(set(hour_for_time_list))
        hour_for_time_list = [hours.index(hour) / len(hours) for hour in hour_for_time_list]
        return np.array(hour_for_time_list).reshape(-1, 1)

    def fit(self, df_threads, y=None):
        return self

    def transform(self, df, y=None):
        # transforms_to_apply = [self.get_weekday, self.get_relative_time_to_election, self.get_hour]
        transforms_to_apply = [self.get_weekday, self.get_hour]
        return np.concatenate(([transform(df[self.key]) for transform in transforms_to_apply]), axis=1)


class TextStatistics:
    """
    Extract features related to sentiment.
    """
    def __init__(self, key):
        """
        :param key: name of tweet text column
        """
        self.features_name = ["sentiment_" + i for i in ["neg", "neu", "pos", "compound"]]
        self.sid = SentimentIntensityAnalyzer()
        self.key = key

    @staticmethod
    def normalize(text):
        text = remove_web_links(text)
        # .translate(string.punctuation)
        text = text.lower()
        return text

    def get_sentiment_analysis(self, texts):
        sentiment_scores = [self.sid.polarity_scores(text) for text in texts]
        sentiment_scores_array = np.array([list(sentiment_score.values()) for sentiment_score in sentiment_scores])
        return sentiment_scores_array

    @staticmethod
    def get_words_count(texts):
        return np.array([len(text.split()) for text in texts]).reshape(-1, 1)

    @staticmethod
    def get_alpha_count(texts):
        return np.array([sum(c.isalpha() for c in text) / len(text) if len(text) > 0 else 0 for text in texts]).reshape(
            -1, 1)

    @staticmethod
    def get_digits_count(texts):
        return np.array([sum(c.isdigit() for c in text) / len(text) if len(text) > 0 else 0 for text in texts]).reshape(
            -1, 1)

    def get_feature_names(self, ):
        return self.features_name

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        transforms_to_apply = [self.get_sentiment_analysis]
        texts = df[self.key].apply(self.normalize)
        return np.concatenate(([transform(texts) for transform in transforms_to_apply]), axis=1)


class WordEmbedder:
    """
    Extract index of word embeddings, according to the word indexes object.
    """
    def __init__(self, key, word_indexes, sequence_len=30):
        """
        :param key: name of tweet text column
        :param word_indexes: dict of form {token: index of token's embedding}
        :param sequence_len: for padding and cropping texts that are too large
        """
        self.features_name = ['w%d' % i for i in range(sequence_len)]
        self.key = key
        self.word_indexes = word_indexes
        self.sequence_len = sequence_len

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        texts = df[self.key]
        X = np.zeros([len(texts), self.sequence_len], dtype=int)
        for i, text in enumerate(texts):
            words = tokenize_for_vectors(text)
            x = [self.word_indexes[word] for word in words if word in self.word_indexes]
            X[i, -len(x):] = x[:self.sequence_len]  # take padding and maximum sequence length into account
        return X


def evaluate_results(clf_name, y_test, predictions, verbose=False):
    """
    Evaluate results and get AUC
    :param clf_name: name of classifier
    :param y_test: test labels
    :param predictions: test predictions
    :param verbose: to print stuff
    :return: test performance in terms of AUC
    """
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
    acc = accuracy_score(y_test, predictions)
    auc = metrics.roc_auc_score(y_test, predictions)
    if verbose:
        print("%s: acc=%.5f, auc=%.5f" % (clf_name, acc, auc))
    return auc
