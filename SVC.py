from preprocessing import TimeStatistics, TextStatistics, ItemSelector
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing as pre
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit

score_index = ['SVC_sigmoid']


def apply_trian_and_test(X_train, y_train, X_test, y_test, score):
    pipeline = Pipeline([('features', FeatureUnion(transformer_list=
    [
        ('tfidf', Pipeline(
            [
                ('selector', ItemSelector(key=pre.processed_tweet_text)),
                ('TfidfVectorizer',
                 TfidfVectorizer(lowercase=False, min_df=1, tokenizer=pre.tokenize, max_features=9000,
                                 ngram_range=(1, 3)))
            ])
         ),
        ('time', TimeStatistics(pre.tweet_time)),
        ('text', TextStatistics(pre.tweet_text)),
    ])), ('clf', SVC(kernel='sigmoid'))])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score[score_index[0]].append(pre.print_evaluation_results(y_test, y_pred))
    return score


def train_and_test_models():
    score = {'SVC_sigmoid': []}
    tweets_df, target = pre.get_raw_data()
    tscv = TimeSeriesSplit(n_splits=2)
    split_counter = 1
    for train_index, test_index in tscv.split(tweets_df):
        print("split counter :" + str(split_counter) + "from " + str(2))
        X_train, X_test = tweets_df.iloc[train_index, :], tweets_df.iloc[test_index, :]
        y_train, y_test = target[train_index], target[test_index]
        apply_trian_and_test(X_train, y_train, X_test, y_test, score)
        split_counter += 1
    print(score_index)
    print(score)


train_and_test_models()
