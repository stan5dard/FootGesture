import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import mean_absolute_error
import time
from torchsummary import summary
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=2)) # 48 * 48

        self.conv_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2))


        self.conv_3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)) # 24 * 24

        self.flatten = nn.Flatten()

    def forward(self, input): # input.shape 2560 * 1 * 32 * 32
        # print('1', input.shape)
        output = self.conv_0(input) # output.shape 2560 * 2 * 32 * 32
        # print('2', output.shape)
        output = self.conv_1(output) # output.shape 2560 * 4 * 16 * 16
        # print('3', output.shape)
        output = self.conv_2(output) # output.shape 2560 * 8 * 8 * 8
        # print('4', output.shape)
        output = self.conv_3(output) # output.shape 2560 * 16 * 4 * 4
        # print('5', output.shape)
        output = self.flatten(output) # output.shape 2560 * 256
        # print('6', output.shape)
        output = output.view(-1, 256) # output.shape 2560 * 256
        # print('7', output.shape)
        return output

class LSTM_CNN(nn.Module):
    def __init__(self, args):
        super(LSTM_CNN, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(64, args.class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        batch_size, timestamps, C, H, W = input.size()
        cnn_in = input.view(batch_size * timestamps, C, H, W) # cnn_in.shape 2560 * 1 * 32 * 32
        # print('0', cnn_in.shape)
        cnn_out = self.cnn(cnn_in) # cnn_out.shape 2560 * 256
        # print('8', cnn_out.shape)
        rnn_in = cnn_out.view(batch_size, timestamps, -1) # rnn_in.shape 64 * 40 * 256
        # print('9', rnn_in.shape)
        rnn_out, (h_n, h_c) = self.rnn(rnn_in) # rnn_out.shape 64 * 40 * 64
        # print('10', rnn_out.shape)
        rnn_out2 = self.linear(rnn_out[:, -1, :]) # rnn_out[:, -1, :].shape 64 * 64
        # rnn_out2.shape 64 * 3
        # print('11', rnn_out[:, -1, :].shape)
        # print('12', rnn_out2.shape)

        return rnn_out2

def train(model, partition, optimizer, loss_fn, args):
    trainloader = DataLoader(partition['train'],
                             batch_size=args.batch_size,
                             shuffle=True, drop_last=True)

    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    total = 0
    correct = 0
    train_loss = 0.0

    for i, (X, y) in enumerate(trainloader):

        X = X.reshape(args.batch_size, args.window_size, 1, 32, 32).float().to(args.device)
        y_true = y.long().to(args.device)
        model.to(args.device)
        model.zero_grad()
        optimizer.zero_grad()

        y_pred = model(X)

        loss = loss_fn(y_pred, y_true.view(-1))
        loss.backward()
        optimizer.step()

        _, y_pred = torch.max(y_pred.data, 1)

        train_loss += loss.item()
        total += y.size(0)

        correct += (y_pred == y_true).sum()
        correct = float(correct)

    train_loss = train_loss / len(trainloader)
    train_acc = 100*correct/total

    return model, train_loss, train_acc

def validate(model, partition, loss_fn, args):
    valloader = DataLoader(partition['val'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)

    model.eval()

    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for i, (X, y) in enumerate(valloader):

            X = X.reshape(args.batch_size, args.window_size, 1, 32, 32).float().to(args.device)
            y_true = y.long().to(args.device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y_true.view(-1))

            _, y_pred = torch.max(y_pred.data, 1)
            val_loss += loss.item()

            total += y.size(0)

            correct += (y_pred == y_true).sum()
            correct = float(correct)

    val_loss = val_loss / len(valloader)
    val_acc = 100 * correct / total

    return val_loss, val_acc

def test(model, partition, args):
    testloader = DataLoader(partition['test'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):

            X = X.reshape(args.batch_size, args.window_size, 1, 32, 32).float().to(args.device)
            y_true = y.long().to(args.device)

            y_pred = model(X)
            _, y_pred = torch.max(y_pred.data, 1)

            total += y.size(0)

            correct += (y_pred == y_true).sum()
            correct = float(correct)

    test_acc = 100 * correct / total

    return test_acc

def experiment(partition, args):

    if args.model == 'LSTM_CNN':
        model = LSTM_CNN(args)
    else:
        raise ValueError('In-valid model choice')

    model.to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # ===================================== #
    # model = nn.DataParallel(model)
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        model, train_loss, train_acc = train(model, partition, optimizer, loss_fn, args)
        val_loss, val_acc = validate(model, partition, loss_fn, args)
        te = time.time()

        # ====== Add Epoch Data ====== #
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        # ============================ #

        print('Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec'.format(epoch+1, train_acc, val_acc, train_loss, val_loss, te - ts))

    test_acc = test(model, partition, args)
    print('##### test_acc', test_acc, '#####')


    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['train_accs'] = train_accs
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc

    # Model Save

    torch.save(model.state_dict(), './trained_model/'+str(time.time()) + 'Op' + args.optim + 'Bat' + str(args.batch_size) + 'Epo' + str(args.epoch) + 'Lr' + str(args.lr).replace('.', '') + 'Win' + str(args.window_size) + 'Model.pt')

    return vars(args), result
