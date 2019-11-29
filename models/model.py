from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn


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
        self.random = RandomForestClassifier(100)

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

    def __init__(self):
        self.random = RandomForestClassifier(100)

    def forward(self, x):
        out = None
        hidden = None

        ##################################################################################
        '''==============================YOUR CODE HERE=============================='''''

        '''=========================================================================='''''
        ##################################################################################

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = None

        ##################################################################################
        '''==============================YOUR CODE HERE=============================='''''

        '''=========================================================================='''''
        ##################################################################################

        return hidden
