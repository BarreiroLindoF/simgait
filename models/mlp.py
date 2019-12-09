import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from model import Cnn
from sklearn.utils import class_weight
from loss import FocalLoss
from statistic_saver import Statistics
from sklearn import preprocessing

print("Loading data")

x = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\extracted\\random_forest_formated\\x.npy")
y = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\extracted\\random_forest_formated\\y.npy")

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)  # transformed string to int

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Transfering data to GPU")

x = torch.from_numpy(x).to(device)
y = torch.from_numpy(y).to(device)

print("Shuffling data")
shuffle_indeces = torch.randperm(x.shape[0])

x = x[shuffle_indeces]
y = y[shuffle_indeces]


print("Preparing train and test sets")
x_train = x[:int(x.shape[0] * 0.8)]
y_train = y[:int(y.shape[0] * 0.8)]

x_test = x[int(x.shape[0] * 0.8):]
y_test = y[int(y.shape[0] * 0.8):]



exit()
print("Intializating model and parameters")

loss = nn.CrossEntropyLoss()
cnn = nn.Sequential(nn.Conv1d(120, 256, 5),
                    # nn.Dropout(0.5),
                    #nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(3),
                    #nn.Conv1d(256, 128, 3),
                    # nn.Dropout(0.5),
                    #nn.BatchNorm1d(128),
                    #nn.LeakyReLU(),
                    #nn.MaxPool1d(3),
                    nn.Conv1d(256, 64, 3),
                    # nn.BatchNorm1d(32),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(2),
                    Cnn.Flatten(),
                    nn.Linear(1984, 128),
                    # nn.Dropout(0.5),
                    #nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 6))
"""
cnn = nn.Sequential(nn.Conv1d(60, 128, 5),
                    # nn.Dropout(0.5),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(3),
                    nn.Conv1d(128, 256, 5),
                    # nn.Dropout(0.5),
                    # nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(3),
                    nn.Conv1d(256, 128, 3),
                    # nn.BatchNorm1d(64),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(128, 128, 2),
                    # nn.BatchNorm1d(64),
                    nn.LeakyReLU(),
                    nn.MaxPool1d(2),
                    Cnn.Flatten(),
                    nn.Linear(128, 128),
                    nn.Dropout(0.5),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 6))"""
# cnn = Cnn()
cnn.to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
print("Starting training")

statistics = Statistics()

n_epochs = 100
batch_size = 32
#loss = FocalLoss()
for epoch in range(n_epochs):

    permutation = torch.randperm(X_train.shape[0])
    for i in range(0, X_train.shape[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        # X_train = torch.from_numpy(X_train).to(device, dtype=torch.float)
        # y_train = torch.from_numpy(y_train).to(device, dtype=torch.long)
        predict = cnn.forward(batch_x)

        output = loss(predict, batch_y)
        output.backward()
        optimizer.step()

        running_loss = output.item()

    cnn.eval()
    predictions = F.softmax(cnn.forward(X_test))
    _, predicted = torch.max(predictions.data, 1)
    # print(predicted.cpu().detach().numpy())
    correct = (predicted == y_test).sum()
    statistics.validation_accuracy.append(correct.cpu().numpy() / y_test.shape[0])
    print('running_loss:', running_loss, "test accuracy", correct.cpu().numpy() / y_test.shape[0], end="\r")
    cnn.train()
statistics.save('cnn_2layer', 'cnns')
