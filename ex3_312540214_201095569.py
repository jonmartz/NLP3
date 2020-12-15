from preprocessing import TimeStatistics, TextStatistics, ItemSelector, WordEmbedder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing as pre
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from RNN import RNNTrumpDetector
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import gzip
import json
import pickle
import pandas as pd


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, criterion, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, 2, bias=True)
        self.output_activation = F.log_softmax
        self.criterion = criterion

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.output_activation(x, -1)
        return x


class FNNClassifier:

    def __init__(self, model, optimizer, criterion, batch_size=256, epochs=40):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, x_train, y_train):
        y_column = np.array(y_train, dtype=int).reshape(-1, 1)
        x_train = x_train.toarray()
        data = np.concatenate([x_train, y_column], axis=1)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        for epoch in range(self.epochs):
            for idx, batch_data in enumerate(loader):
                self.optimizer.zero_grad()
                batch_features = batch_data[:, :-1].float()
                batch_y = batch_data[:, -1].long()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, x_test):
        x_test = x_test.toarray()
        data = torch.from_numpy(x_test).float()
        self.optimizer.zero_grad()
        outputs = self.model(data)
        return outputs


def create_fnn_model(features_count, hidden_layer_size=300, learn_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    net = Net(input_size=features_count, hidden_size=hidden_layer_size, criterion=criterion, num_classes=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)
    return FNNClassifier(model=net, optimizer=optimizer, criterion=criterion)


def create_rnn_model(n_features):
    model = RNNTrumpDetector(word_vectors, word_indexes, sequence_len=30, n_features=n_features, lstm_out_dim=256,
                             lstm_layers=2, dense_layer_dims=[], epochs=20, device=device, n_batches=10,
                             lr=0.005, lstm_dropout=0.2, lstm_out_dropout=0.2)
    model.to(device)
    return model


def create_pipeline(clf_name, clf):
    transformers = [
        ('tfidf', Pipeline(
            [
                ('selector', ItemSelector(key=pre.processed_tweet_text)),
                ('TfidfVectorizer', TfidfVectorizer(lowercase=False, min_df=1, tokenizer=pre.tokenize,
                                                    max_features=9000, ngram_range=(1, 3)))
            ])
         ),
        ('time', TimeStatistics(pre.tweet_time)),
        ('text', TextStatistics(pre.tweet_text))
    ]
    if clf_name == 'RNN':
        transformers.insert(0, ('embeddings', WordEmbedder(pre.tweet_text, clf.word_indexes)))
    return Pipeline([('features', FeatureUnion(transformers)), ('clf', clf)])


def apply_train_and_test(X_train, y_train, X_test, y_test, score, verbose=False):
    for clf_name, item in classifiers.items():
        clf_class, params = item
        clf = clf_class(**params)
        pipeline = create_pipeline(clf_name, clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if clf_name == 'FFNN':
            y_pred = list(map(lambda x: 0 if x[0] > x[1] else 1, y_pred))
        score[clf_name].append(pre.evaluate_results(clf_name, y_test, y_pred, verbose))


def train_best_model(n_splits=10, verbose=True):
    score = {clf_name: [] for clf_name in classifiers.keys()}
    df, target = pre.get_raw_data('trump_train.tsv')

    # cross-validate to find best model
    cv = KFold(n_splits=n_splits, shuffle=True)
    split_idx = 0
    for train_index, test_index in cv.split(df):
        split_idx += 1
        if verbose:
            print("\nfold %d/%d" % (split_idx, n_splits))
        X_train, X_test = df.iloc[train_index, :], df.iloc[test_index, :]
        y_train, y_test = target[train_index], target[test_index]
        apply_train_and_test(X_train, y_train, X_test, y_test, score, verbose)
    avg_score = {name: np.mean(s) for name, s in score.items()}
    pd.DataFrame(score).to_csv('results.csv', index_label='fold')

    # train best model on full dataset
    if verbose:
        print('\naverage scores:')
        for name, auc in avg_score.items():
            print('\t%s: auc=%.5f' % (name, auc))
        print('\nfitting best model on full train set...')
    best_model_name = max(avg_score)
    best_model_class, best_params = classifiers[best_model_name]
    pipeline = create_pipeline(best_model_name, best_model_class(**best_params))
    pipeline.fit(df, target)
    with open('best_model.pickle', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pipeline


def load_best_model():
    with open('best_model.pickle', 'rb') as handle:
        return pickle.load(handle)


def predict(m, fn):
    df, target = pre.get_raw_data(fn)
    return m.predict(df).astype(int)  # todo: return as list? As instructions say


torch.manual_seed(0)  # for reproducibility

classifiers = {
    'Logistic_Regression': (LogisticRegression, {'solver': 'liblinear'}),
    'SVC_sigmoid': (SVC, {'kernel': 'sigmoid'}),
    'SVC_rbf': (SVC, {'kernel': 'rbf'}),
    'SVC_linear': (SVC, {'kernel': 'linear'}),
    'FFNN': (create_fnn_model, {'features_count': 9006}),
    'RNN': (create_rnn_model, {'n_features': 9006}),
    'AdaBoost': (AdaBoostClassifier, {'n_estimators': 100}),
}

if 'RNN' in classifiers:
    # load word vectors
    with gzip.open('vectors.sav', 'r') as file:
        json_bytes = file.read()
    word_vectors = json.loads(json_bytes.decode('utf-8'))
    word_vectors = np.array([[0] * len(word_vectors[0])] + word_vectors)  # add padding
    with gzip.open('vocab.sav', 'r') as file:
        json_bytes = file.read()
    vocabulary = ['-'] + json.loads(json_bytes.decode('utf-8'))  # the '-' is for padding
    word_indexes = {word: i for i, word in enumerate(vocabulary)}

    # look for GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

if __name__ == "__main__":
    m = train_best_model()
    # m = load_best_model()

    # save test predictions
    y_pred = predict(m, 'trump_test.tsv')
    with open('312540214_201095569.txt', 'w') as file:
        y_string = str(y_pred)[1:-1].replace('\n', '')
        file.write(y_string)
    print(y_string)

    print('\ndone')
