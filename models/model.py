from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
from torch.functional import F


# Abstract class simulating and interface
# It is useful for any polymorphism operations

# Also, any class may be added to this file using any library or method
# the objective is to simplify the testing of the models
class SupervisedModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


# Concrete class implementing kNN algorithm using sklearn library
class Knn(SupervisedModel):

    def __init__(self, neighbors):
        self.knn = KNeighborsClassifier(n_neighbors=neighbors)

    def fit(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        return self.knn.predict(X)


# Concrete class implementing DecisionTree algorithm using sklearn library
class DecisionTree(SupervisedModel):

    def __init__(self):
        self.tree = DecisionTreeClassifier()

    def fit(self, X, y):
        self.tree.fit(X, y)

    def predict(self, X):
        return self.tree.predict(X)


# Concrete class implementing SVM algorithm using sklearn library
class SVM(SupervisedModel):

    def __init__(self):
        self.svm = SVC(gamma='scale')

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)


# Concrete class implementing RandomForest algorithm using sklearn library
class RandomForest(SupervisedModel):

    def __init__(self):
        self.random = RandomForestClassifier(1)

    def fit(self, X, y):
        self.random.fit(X, y)

    def predict(self, X):
        return self.random.predict(X)


# Concrete class implementing Recurrent Neural Network using PyTorch
class Rnn(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_dim: [int, int], n_layers: int):
        # Define parameters
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(x.size(0))

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, input_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, input_size, self.hidden_dim)

        return hidden


# Concrete class implementing Convolutional Neural Network using PyTorch
class Cnn(nn.Module):

    class Flatten(nn.Module):
        def forward(self, input):
            self.result = input.view(input.size(0), -1)
            return self.result

    def __init__(self):
        super(Cnn, self).__init__()

        # top conv
        self.conv1 = nn.Conv1d(60, 64, 3) # from 60 channels with 64 filters, conv kernel 3
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool1d(2)
        self.conv12 = nn.Conv1d(64, 64, 3)
        self.relu12 = nn.ReLU()
        self.mp12 = nn.MaxPool1d(2)

        # middle conv
        self.conv2 = nn.Conv1d(60, 64, 5) # from 60 channels with 64 filters, conv kernel 3
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool1d(2)
        self.conv22 = nn.Conv1d(64, 64, 5)
        self.relu22 = nn.ReLU()
        self.mp22 = nn.MaxPool1d(2)

        # middle conv
        self.conv3 = nn.Conv1d(60, 64, 7) # from 60 channels with 64 filters, conv kernel 3
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool1d(2)
        self.conv23 = nn.Conv1d(64, 64, 7)
        self.relu23 = nn.ReLU()
        self.mp23 = nn.MaxPool1d(2)

        # concat
        self.flatten = self.Flatten()

        # convolution all image

        self.conv_all = nn.Conv1d(64, 32, 3)
        self.mp_all = nn.MaxPool1d(2)
        self.relu_all = nn.ReLU()

        # Flatten

        self.fc1 = nn.Linear(1000, 6)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn12 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn21 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn31 = nn.BatchNorm1d(64)

    def forward(self, x):
        # top conv
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)
        z = self.mp1(z)
        z = self.conv12(z)
        z = self.bn12(z)
        z = self.relu12(z)
        z = self.mp12(z)

        # middle conv
        y = self.conv2(x) # from 60 channels with 64 filters, conv kernel 3
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.mp2(y)
        y = self.conv22(y)
        y = self.bn21(y)
        y = self.relu22(y)
        y = self.mp22(y)

        # middle conv
        w = self.conv3(x) # from 60 channels with 64 filters, conv kernel 3
        w = self.bn3(w)
        w = self.relu3(w)
        w = self.mp3(w)
        w = self.conv23(w)
        w = self.bn31(w)
        w = self.relu23(w)
        w = self.mp23(w)

        # concat

        x = torch.cat((z, y, w), dim=2) # horizontal concat

        # convolution all image

        x = self.conv_all(x)
        x = self.mp_all(x)
        x = self.relu_all(x)

        # Flatten

        x = self.flatten(x)
        self.fc1 = nn.Linear(self.flatten.result.shape[1], 6)

        return x

    def init_hidden(self, batch_size):
        hidden = None

        ##################################################################################
        '''==============================YOUR CODE HERE=============================='''''

        '''=========================================================================='''''
        ##################################################################################

        return hidden
