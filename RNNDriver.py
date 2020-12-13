import gzip
import json
import numpy as np
import pandas as pd
from preprocessing import tokenize
from sklearn.model_selection import KFold
import torch
from RNN import RNNTrumpDetector, train
torch.manual_seed(1)

sequence_len = 30  # only ~100 texts have more than 30 tokens
num_folds = 5  # for cross-validation
hidden_dim = 512  # for the LSTM layer
n_layers = 2  # num of LSTM layers stacked on top of each other
epochs = 10
batch_size = 256

print('loading word vectors...')
with gzip.open('vectors.sav', 'r') as file:
    json_bytes = file.read()
word_vectors = json.loads(json_bytes.decode('utf-8'))
word_vectors = np.array([[0] * len(word_vectors[0])] + word_vectors)  # add padding
with gzip.open('vocab.sav', 'r') as file:
    json_bytes = file.read()
vocabulary = ['-'] + json.loads(json_bytes.decode('utf-8'))  # the '-' is for padding
word_indexes = {word: i for i, word in enumerate(vocabulary)}

print('loading training and test sets...')
df_train, df_test = pd.read_csv('dataset_train.csv'), pd.read_csv('dataset_test.csv')
texts_train, texts_test = df_train['tweet text'], df_test['tweet text']
y = df_train['label'].to_numpy()
X_train_and_valid = np.zeros([len(texts_train), sequence_len], dtype=int)
X_test = np.zeros([len(texts_test), sequence_len], dtype=int)
for texts, X in zip([texts_train, texts_test], [X_train_and_valid, X_test]):
    for i, text in enumerate(texts):
        words = tokenize(text)
        x = [word_indexes[word] for word in words]
        X[i, -len(x):] = x[:sequence_len]  # take padding and maximum sequence length into account

print('starting k-fold cross-validation...')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
folds = list(KFold(n_splits=num_folds, shuffle=True, random_state=1).split(X_train_and_valid))
for fold_idx, fold in enumerate(folds):
    print('\nfold %d/%d' % (fold_idx + 1, num_folds))
    indexes_train, indexes_valid = fold
    X_train, y_train = X_train_and_valid[indexes_train], y[indexes_train]
    X_valid, y_valid = X_train_and_valid[indexes_valid], y[indexes_valid]

    model = RNNTrumpDetector(word_vectors, hidden_dim, n_layers, device)
    model.to(device)
    train(model, X_train, y_train, X_valid, y_valid, epochs, batch_size, device)

print('done')
