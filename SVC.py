from preprocessing import TimeStatistics, TextStatistics, ItemSelector, WordEmbedder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocessing as pre
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from RNN import RNNTrumpDetector
from gensim.models import KeyedVectors
import gzip
import json


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


def create_rnn_model(n_features, lstm_out_dim=512, lstm_layers=2, epochs=5, n_batches=30, sequence_len=30,
                     lr=0.001, dense_layer_dims=[1024, 512, 256, 128]):
    # loading word vectors
    with gzip.open('vectors.sav', 'r') as file:
        json_bytes = file.read()
    word_vectors = json.loads(json_bytes.decode('utf-8'))
    word_vectors = np.array([[0] * len(word_vectors[0])] + word_vectors)  # add padding
    with gzip.open('vocab.sav', 'r') as file:
        json_bytes = file.read()
    vocabulary = ['-'] + json.loads(json_bytes.decode('utf-8'))  # the '-' is for padding
    word_indexes = {word: i for i, word in enumerate(vocabulary)}

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = RNNTrumpDetector(word_vectors, word_indexes, sequence_len, n_features, lstm_out_dim, lstm_layers,
                             dense_layer_dims, epochs, device, n_batches=n_batches, lr=lr)
    model.to(device)
    return model


classifiers = {
    # 'Logistic_Regression': LogisticRegression(solver='liblinear'),
    # 'SVC_sigmoid': SVC(kernel='sigmoid'),
    # 'SVC_rbf': SVC(kernel='rbf'),
    # 'SVC_linear': SVC(kernel='linear'),
    # 'FFNN': create_fnn_model(9006),
    'RNN': create_rnn_model(9006)
}


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


def apply_train_and_test(X_train, y_train, X_test, y_test, score):
    for clf_name, clf in classifiers.items():
        pipeline = create_pipeline(clf_name, clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if clf_name == 'FFNN':
            y_pred = list(map(lambda x: 0 if x[0] > x[1] else 1, y_pred))
        score[clf_name].append(pre.print_evaluation_results(y_test, y_pred))


def train_and_test_models():
    score = {clf_name: [] for clf_name in classifiers.keys()}
    tweets_df, target = pre.get_raw_data()
    tscv = TimeSeriesSplit(n_splits=2)
    split_counter = 1
    for train_index, test_index in tscv.split(tweets_df):
        print("split counter :" + str(split_counter) + "from " + str(2))
        X_train, X_test = tweets_df.iloc[train_index, :], tweets_df.iloc[test_index, :]
        y_train, y_test = target[train_index], target[test_index]
        apply_train_and_test(X_train, y_train, X_test, y_test, score)
        split_counter += 1
    print(score)


train_and_test_models()
