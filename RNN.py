import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


class RNNTrumpDetector(nn.Module):
    def __init__(self, word_vectors, hidden_dim, n_layers, device, lstm_dropout=0.5, dense_dropout=0.2):
        super(RNNTrumpDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_vectors), freeze=True)
        self.lstm = nn.LSTM(word_vectors.shape[1], hidden_dim, n_layers, dropout=lstm_dropout, batch_first=True)
        self.dropout = nn.Dropout(dense_dropout)
        self.dense = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.dense(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden


def train(model, X_train, y_train, X_valid, y_valid, epochs, batch_size, device,
          lr=0.005, evaluate_every=2, clip=5):

    # load data
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

    rnd_valid_acc = float(np.mean(y_valid))
    rnd_valid_acc = max(rnd_valid_acc, 1 - rnd_valid_acc)

    total_train_losses = []
    total_valid_losses = []
    total_valid_accs = []

    # optimizer and loss function
    loss_function = torch.nn.BCELoss()  # binary cross-entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    valid_loss_min = np.Inf
    model.train()
    counter = 0
    for epoch in range(epochs):
        h = model.init_hidden(batch_size)

        batch = 0
        for inputs, labels in train_loader:
            batch += 1
            if batch == len(train_loader):
                continue  # todo: fix last batch instead of skipping it
            counter += 1
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, h = model(inputs, h)
            loss = loss_function(output.squeeze(), labels.float())

            # pred = torch.round(output.squeeze())
            # correct_tensor = pred.eq(labels.float().view_as(pred))
            # correct = np.squeeze(correct_tensor.cpu().numpy())
            # train_acc = np.mean(correct)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % evaluate_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                valid_accs = []
                model.eval()
                j = 0
                for inp, lab in val_loader:
                    j += 1
                    if j == len(val_loader):
                        continue
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h)
                    val_loss = loss_function(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                    pred = torch.round(out.squeeze())
                    correct_tensor = pred.eq(labels.float().view_as(pred))
                    correct = np.squeeze(correct_tensor.cpu().numpy())
                    valid_accs.append(np.mean(correct))

                model.train()
                val_losses_mean = float(np.mean(val_losses))
                val_accs_mean = float(np.mean(valid_accs))
                # print("Epoch: {}/{}...".format(i + 1, epochs),
                #       "Step: {}...".format(counter),
                #       "Loss: {:.6f}...".format(loss.item()),
                #       "Val Loss: {:.6f}".format(np.mean(val_losses)))
                best_str = ''
                if np.mean(val_losses) <= valid_loss_min:
                    # torch.save(model.state_dict(), './state_dict.pt')
                    valid_loss_min = val_losses_mean
                    best_str = ' NEW BEST'
                print('epoch:%d/%d batch:%d/%d train_loss:%.5f valid_loss:%.5f valid_acc:%.5f rnd_valid_acc=%.5f%s' %
                      (epoch + 1, epochs, batch, len(train_loader), loss.item(), val_losses_mean,
                       val_accs_mean, rnd_valid_acc, best_str))
                total_train_losses.append(loss.item())
                total_valid_losses.append(val_losses_mean)
                total_valid_accs.append(val_accs_mean)
    x = list(range(len(total_train_losses)))
    plt.plot(x, total_train_losses, label='train')
    plt.plot(x, total_valid_losses, label='valid')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.plot(x, total_valid_accs, label='valid')
    plt.plot([x[0], x[-1]], [rnd_valid_acc, rnd_valid_acc], 'k--', label='random')
    plt.ylabel('acc')
    plt.legend()
    plt.show()


def predict(model, X, device):
    
    data = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(data)

    # loss_function = torch.nn.BCELoss()  # binary cross-entropy
    # losses = []
    # num_correct = 0
    h = model.init_hidden(1)

    model.eval()
    # for inputs in loader:

    # inputs = next(loader)
    inputs = loader[0]

    h = tuple([each.data for each in h])
    # inputs, labels = inputs.to(device), labels.to(device)
    inputs = inputs.to(device)
    output, h = model(inputs, h)
    # loss = loss_function(output.squeeze(), labels.float())
    # losses.append(loss.item())
    pred = torch.round(output.squeeze())  # rounds the output to 0/1
    # correct_tensor = pred.eq(labels.float().view_as(pred))
    # correct = np.squeeze(correct_tensor.cpu().numpy())
    # num_correct += np.sum(correct)

    # print("loss: {:.3f}".format(np.mean(losses)))
    # acc = num_correct / len(loader.dataset)
    # print("accuracy: {:.3f}%".format(acc * 100))

    return pred