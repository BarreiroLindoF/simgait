import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.autograd import Variable
import numpy as np
import sys
from statistic_saver import Statistics
import os

# Project directory to make this code runnable on any windows system (to be changed on mac)
project_dir = os.path.expanduser(os.path.dirname(os.getcwd()))

# Import markers class to be enable to read data stored in npy files (array of Markers object)
sys.path.append(project_dir + "\\models_preprocess\\")
from rnn_preprocess import Markers

torch.autograd.set_detect_anomaly(True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_units, nb_layer, nb_labels):
        super(RNN, self).__init__()
        self.num_layers = nb_layer
        self.hidden_dim = hidden_units

        self.rnn = nn.RNN(           
            input_size=input_size,    # Size of the input sequence (None if dynamic)
            hidden_size=hidden_units, # rnn hidden unit
            num_layers=nb_layer,      # number of rnn layer
            batch_first=True,         # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=0.5,
            nonlinearity='relu'
        )

        self.out = nn.Linear(hidden_units, nb_labels)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())
                
        # Index hidden state of last time step
        # out[:, -1, :] just want last time step hidden states
        out = self.out(out[:, -1, :])

        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_units, nb_layer, nb_labels):
        super(GRU, self).__init__()
        self.num_layers = nb_layer
        self.hidden_dim = hidden_units
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_dim, dropout=0.5,
                          batch_first=True)
        self.out = nn.Linear(hidden_units, nb_labels)

    def forward(self, x):
        out, hidden = self.gru(x)
        out = self.out(out[:, -1, :])

        return out


class LSTM (nn.Module):
    def __init__(self, input_size, hidden_units, nb_layer, nb_labels):
        super(LSTM, self).__init__()
        self.num_layers = nb_layer
        self.hidden_dim = hidden_units

        self.lstm = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_units,  # rnn hidden unit
            num_layers=nb_layer,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=0.5
        )

        self.out = nn.Linear(hidden_units, nb_labels)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()

        # 60 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.out(out[:, -1, :]) 

        return out

"""
# Load data with fixed size (simple array numpy)
#########################################
X_train = Variable(torch.tensor(np.load("C:\\Users\\Lucas\\Desktop\\gaitmasteris\\data\\rnn_formated\\x_train_norm.npy", allow_pickle=True)))
X_test = Variable(torch.tensor(np.load("C:\\Users\\Lucas\\Desktop\\gaitmasteris\\data\\rnn_formated\\x_test_norm.npy")))
#########################################
"""

# Load data with dynamic array size (numpy array of Markers object in which the array size contained is dynamic)
#########################################
# Get all directory names for cross validation
dir_names = os.listdir(project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\")
dir_names = filter(lambda k: 'CV' in k, dir_names)

# Define hyper-parameters
hidden_units = 256
nb_layer = 1
lr = 1e-04       # learning rate
nb_epoch = 100
batch_size = 512
time_step = 60   # rnn time step - here this represents that the RNN would be able to keep in memory the the 60 sensors data



X_train_npy = np.load("C:\\Users\\Lucas\\Desktop\\gaitmasteris\\data\\rnn_formated\\x_train_norm_dynamic.npy", allow_pickle=True)
X_test_npy = np.load("C:\\Users\\Lucas\\Desktop\\gaitmasteris\\data\\rnn_formated\\x_test_norm_dynamic.npy", allow_pickle=True)

y_train = np.load("C:\\Users\\Lucas\\Desktop\\gaitmasteris\\data\\rnn_formated\\y_train.npy")
y_test = np.load("C:\\Users\\Lucas\\Desktop\\gaitmasteris\\data\\rnn_formated\\y_test.npy")

# As we have multiple arrays with differents length and Pytorch must have the same length to do batch, we will padd with zeros the smaller ones
# get the length of each seq in datasets
seq_lengths_train = torch.LongTensor([seq[0].getMarkers().shape[1] for seq in X_train_npy]).cuda()
seq_lengths_test = torch.LongTensor([seq[0].getMarkers().shape[1] for seq in X_test_npy]).cuda()

nb_sensors = X_train_npy[0][0].getMarkers().shape[0]

# Create a tensor for each dataset with the shape of the max length sequence with zeros every where
seq_train_tensor = torch.zeros((len(X_train_npy), nb_sensors, seq_lengths_train.max())).cuda()
seq_test_tensor = torch.zeros((len(X_test_npy), nb_sensors, seq_lengths_train.max())).cuda()

# Fill the zeros tensors with sequences values (each sequence with a length smaller than the max seq lenght will have some 0 padding at the end) 
for i in range(X_train_npy.shape[0]):
    curr_seq_length = seq_lengths_train[i].item()
    seq_train_tensor[i, :, 0:curr_seq_length] = torch.FloatTensor(X_train_npy[i][0].getMarkers())

for i in range(X_test_npy.shape[0]):
    curr_seq_length = seq_lengths_test[i].item()
    seq_test_tensor[i, :, 0:curr_seq_length] = torch.FloatTensor(X_test_npy[i][0].getMarkers())

#########################################
"""
X_train = []
for i in range(X_train_npy.shape[0]):
    print(i)
    X_train.append(torch.Tensor(X_train_npy[i][0].getMarkers()))
"""

# Define some hyper parameters
input_size = seq_lengths_train.max()   # rnn input size / nb frames
nb_labels = y_train.shape[1]

# Instantiate our NN
rnn = RNN(input_size, hidden_units, nb_layer, nb_labels)
rnn.cuda()

# Define an optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)  # optimize all nn parameters

# Define the loss function
y_train = torch.tensor(np.argmax(y_train, axis=1)) # Get label no one hot encoded for y as Loss of pytorch need it like that
y_test = np.argmax(y_test, axis=1)
loss_func = nn.CrossEntropyLoss()

# Create a statistic object to store informations during the training 
stats = Statistics()

# Data Loader for easy mini-batch return in training
train_loader = DataLoader(dataset=seq_train_tensor, batch_size=batch_size, shuffle=False)

for epoch in range(nb_epoch):
    idx_start = 0
    idx_end = 0
    
    #scheduler.step()
    
    for step, x in enumerate(train_loader):        # gives batch data
        rnn.train()
        idx_end += x.shape[0]

        # reshape x to (batch, time_step, input_size)
        b_x = x.view(-1, time_step, input_size).cuda()            
        b_y = Variable(y_train[idx_start:idx_end]).cuda()  
                        
        # rnn output
        output = rnn(b_x).cuda()
                
        # cross entropy loss
        loss = loss_func(output, b_y)
        
         # clear gradients for this training step
        optimizer.zero_grad()
        
        # backpropagation, compute gradients
        loss.backward()
        
        # Updates parameters
        optimizer.step()
        
        if step % 13 == 0:
            pred_y = torch.max(output, 1)[1].cpu().data.numpy().squeeze()
            accuracy = sum(pred_y == y_train[idx_start: idx_end].data.numpy()) / y_train[idx_start: idx_end].data.numpy().size
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| train accuracy : ', accuracy)            
            stats.training_accuracy.append(accuracy)
            
            rnn.eval()
            x_test = seq_test_tensor.view(-1, time_step, input_size)
            test_output = rnn(x_test) # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()
            
            accuracy = sum(pred_y == y_test) / float(y_test.size)
            stats.validation_accuracy.append(accuracy)
            print('Test accuracy: %.2f' % accuracy)    
        
        idx_start += x.shape[0]

stats.model_structure = "GRU with dropout 0.5, timestep 60, hidden units 256, nb layer 1, nb epoch 100, batch size 512, lr 1e-04"

stats.save("gru", "C:\\Users\\lucas\\Desktop\\gaitmasteris\\models\\rnn_stats")
"""
rnn.eval()
test_output = rnn(X_test.cuda()) # (samples, time_step, input_size)
pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()
accuracy = sum(pred_y == y_test) / float(y_test.size)
print('Test accuracy: %.2f' % accuracy)       

print(y_train[y_train == 3].size()[0] / y_train.size()[0])
"""