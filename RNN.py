import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


class RNNTrumpDetector(nn.Module):
    def __init__(self, word_vectors, word_indexes, sequence_len, n_features, lstm_out_dim, lstm_layers,
                 dense_layer_dims, epochs, device, lstm_dropout=0.2, lstm_out_dropout=0.2, batch_size=128,
                 lr=0.005, steps_between_validations=1, clip_size=5):
        super(RNNTrumpDetector, self).__init__()
        self.word_indexes = word_indexes
        self.sequence_len = sequence_len
        self.n_features = n_features
        self.lstm_layers = lstm_layers
        self.lstm_out_dim = lstm_out_dim
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.steps_between_validations = steps_between_validations
        self.clip_size = clip_size

        # from input to lstm output
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_vectors), freeze=True)
        self.lstm = nn.LSTM(word_vectors.shape[1], lstm_out_dim, lstm_layers, dropout=lstm_dropout, batch_first=True)
        self.lstm_out_dropout = nn.Dropout(lstm_out_dropout)

        # from dense layers to final output
        prev_dim = lstm_out_dim + n_features
        self.dense_layers = []
        for next_dim in dense_layer_dims:
            self.dense_layers.append([nn.Linear(prev_dim, next_dim), nn.ReLU()])
            prev_dim = next_dim
        self.last_dense = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # loss and optimizer
        self.loss_function = torch.nn.BCELoss()  # binary cross-entropy
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        x_embedding, x_features = x[:, :self.sequence_len], x[:, self.sequence_len:]
        embeddings = self.embedding(x_embedding)
        lstm_out, hidden = self.lstm(embeddings, hidden)
        # lstm_out = lstm_out.contiguous().view(-1, self.lstm_out_dim)
        lstm_out = lstm_out[:, -1, :]  # keep only last lstm state
        lstm_out = self.lstm_out_dropout(lstm_out)
        out = torch.cat((lstm_out, x_features), 1)
        for dense, relu in self.dense_layers:
            out = dense(out)
            out = relu(out)
        out = self.last_dense(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_out_dim).zero_().to(self.device),
                  weight.new(self.lstm_layers, batch_size, self.lstm_out_dim).zero_().to(self.device))
        return hidden

    def fit(self, X_train, y_train, valid_frac=0.2, verbose=True, plot_history=True):

        X_train, y_train = X_train.toarray(), np.array(y_train, dtype=int)  # .reshape(-1, 1)

        validate = valid_frac > 0
        if validate:
            train_len = int(len(X_train) * (1 - valid_frac))
            X_valid = X_train[train_len:]
            y_valid = y_train[train_len:]
            X_train = X_train[:train_len]
            y_train = y_train[:train_len]
            val_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
            val_loader = DataLoader(val_data, shuffle=True, batch_size=self.batch_size)

            total_valid_losses = []
            total_valid_accs = []
            valid_loss_min = np.Inf

        train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)

        total_train_losses = []

        self.train()
        counter = 0
        for epoch in range(self.epochs):
            h = self.init_hidden(self.batch_size)

            batch = 0
            for x_train, y_train in train_loader:
                batch += 1
                if batch == len(train_loader):
                    continue  # todo: fix last batch instead of skipping it
                counter += 1
                h = tuple([e.data for e in h])
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                self.zero_grad()
                output, h = self(x_train, h)
                loss = self.loss_function(output.squeeze(), y_train.float())

                # pred = torch.round(output.squeeze())
                # correct_tensor = pred.eq(labels.float().view_as(pred))
                # correct = np.squeeze(correct_tensor.cpu().numpy())
                # train_acc = np.mean(correct)

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_size)
                self.optimizer.step()

                if validate and counter % self.steps_between_validations == 0:
                    val_h = self.init_hidden(self.batch_size)
                    val_losses = []
                    valid_accs = []
                    self.eval()
                    j = 0
                    for x_valid, y_valid in val_loader:
                        j += 1
                        if j == len(val_loader):
                            continue
                        val_h = tuple([each.data for each in val_h])
                        x_valid, y_valid = x_valid.to(self.device), y_valid.to(self.device)
                        out, val_h = self(x_valid, val_h)
                        val_loss = self.loss_function(out.squeeze(), y_valid.float())
                        val_losses.append(val_loss.item())

                        pred = torch.round(out.squeeze())
                        correct_tensor = pred.eq(y_train.float().view_as(pred))
                        correct = np.squeeze(correct_tensor.cpu().numpy())
                        valid_accs.append(np.mean(correct))

                    self.train()
                    val_losses_mean = float(np.mean(val_losses))
                    val_accs_mean = float(np.mean(valid_accs))
                    best_str = ''
                    if np.mean(val_losses) <= valid_loss_min:
                        # torch.save(model.state_dict(), './state_dict.pt')
                        valid_loss_min = val_losses_mean
                        best_str = ' NEW BEST'
                    if verbose:

                        # for comparing validation performance with a random flip coin classifier
                        rnd_valid_acc = float(np.mean(y_train))
                        rnd_valid_acc = max(rnd_valid_acc, 1 - rnd_valid_acc)

                        print(
                            'epoch:%d/%d batch:%d/%d train_loss:%.5f valid_loss:%.5f valid_acc:%.5f rnd_valid_acc=%.5f%s' %
                            (epoch + 1, self.epochs, batch, len(train_loader), loss.item(), val_losses_mean,
                             val_accs_mean, rnd_valid_acc, best_str))
                    total_train_losses.append(loss.item())
                    total_valid_losses.append(val_losses_mean)
                    total_valid_accs.append(val_accs_mean)

        x = list(range(len(total_train_losses)))
        if plot_history:
            plt.plot(x, total_train_losses, label='train')
            if validate:
                plt.plot(x, total_valid_losses, label='valid')
            plt.ylabel('loss')
            plt.legend()
            if validate:
                plt.show()
                plt.plot(x, total_valid_accs, label='valid')
                plt.plot([x[0], x[-1]], [rnd_valid_acc, rnd_valid_acc], 'k--', label='random')
                plt.ylabel('acc')
                plt.legend()
                plt.show()

    def predict(self, X):
        X = X.toarray()
        data = torch.from_numpy(X).float()
        self.optimizer.zero_grad()
        outputs = self(data)
        return torch.round(outputs.squeeze())
