import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from models.model import Cnn
print("Loading data")
X_train = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\normalized\\CV1\\1d_X_train.npy")
y_train = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\normalized\\CV1\\1d_y_train.npy")
y_train = np.argmax(y_train, axis=1)
X_test = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\normalized\\CV1\\1d_X_test.npy")
y_test = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\normalized\\CV1\\1d_y_test.npy")
y_test = np.argmax(y_test, axis=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Transfering data to GPU")
X_train = torch.from_numpy(X_train).to(device, dtype=torch.float)
y_train = torch.from_numpy(y_train).to(device, dtype=torch.long)
X_test = torch.from_numpy(X_test).to(device, dtype=torch.float)
y_test = torch.from_numpy(y_test).to(device, dtype=torch.long)


# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)

print("Intializating model and parameters")

loss = nn.CrossEntropyLoss()
cnn = nn.Sequential(nn.Conv1d(60, 64, 5),
                    nn.Dropout(0.5),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 64, 3),
                    nn.Dropout(0.5),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 64, 3),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    Cnn.Flatten(),
                    nn.Linear(640, 256),
                    nn.Dropout(0.5),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, 6))
# cnn = Cnn()
cnn.to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
print("Starting training")
for epoch in range(1000):

    optimizer.zero_grad()
    predict = cnn.forward(X_train)
    output = loss(predict, y_train)
    output.backward()
    optimizer.step()

    running_loss = output.item()


    if epoch % 50 == 0:
        print('running_loss:', running_loss)
        cnn.eval()
        predictions = F.softmax(cnn.forward(X_test))
        _, predicted = torch.max(predictions.data, 1)
        correct = (predicted == y_test).sum()
        print(correct.cpu().numpy() / y_test.shape[0])
        cnn.train()
