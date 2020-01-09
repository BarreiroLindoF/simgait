import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.autograd import Variable
import numpy as np
import sys
from statistic_saver import Statistics, CrossValStatistics
import os
import argparse

# Project directory to make this code runnable on any windows system (to be changed on mac)
project_dir = os.path.expanduser(os.path.dirname(os.getcwd()))

# Import markers class to be enable to read data stored in npy files (array of Markers object)
sys.path.append(project_dir + "\\extraction\\models_preprocess\\")
from rnn_preprocess import Markers

torch.autograd.set_detect_anomaly(True)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_units, nb_layer, nb_labels):
        super(RNN, self).__init__()
        self.num_layers = nb_layer
        self.hidden_dim = hidden_units

        self.rnn = nn.RNN(
            input_size=input_size,  # Size of the input sequence (None if dynamic)
            hidden_size=hidden_units,  # rnn hidden unit
            num_layers=nb_layer,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
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


class LSTM(nn.Module):
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


def get_dynamic_dataset(directory):
    # Load train and test data (matrix of objects --> Markers)
    X_train_npy = np.load(
       directory + "\\x_train_norm_dynamic.npy",
        allow_pickle=True)
    X_test_npy = np.load(
        directory + "\\x_test_norm_dynamic.npy",
        allow_pickle=True)

    # Load labels for train and test (numpy array)
    y_train = np.load(directory + "\\y_train.npy")
    y_test = np.load(directory + "\\y_test.npy")

    # get lengths of each seq in train and test
    seq_lengths_train = torch.LongTensor([seq[0].getMarkers().shape[1] for seq in X_train_npy]).cuda()
    seq_lengths_test = torch.LongTensor([seq[0].getMarkers().shape[1] for seq in X_test_npy]).cuda()

    if seq_lengths_train.max() > seq_lengths_test.max():
        max_seq_length = seq_lengths_train.max()
    else:
        max_seq_length = seq_lengths_test.max()

    nb_sensors = X_train_npy[0][0].getMarkers().shape[0]

    # Create a tensor for each dataset with the shape of the max length sequence with zeros everywhere
    x_train = torch.zeros((len(X_train_npy), nb_sensors, max_seq_length)).cuda()
    x_test = torch.zeros((len(X_test_npy), nb_sensors, max_seq_length)).cuda()

    # Fill the zeros tensors with sequences values (each sequence with a length smaller than the max seq lenght will have some 0 padding at the end)
    for i in range(X_train_npy.shape[0]):
        curr_seq_length = seq_lengths_train[i].item()
        x_train[i, :, 0:curr_seq_length] = torch.FloatTensor(X_train_npy[i][0].getMarkers())

    for i in range(X_test_npy.shape[0]):
        curr_seq_length = seq_lengths_test[i].item()
        x_test[i, :, 0:curr_seq_length] = torch.FloatTensor(X_test_npy[i][0].getMarkers())

    return x_train, y_train, x_test, y_test, max_seq_length


def get_static_dataset(directory):
    # Load data with fixed size (simple array numpy)
    x_train = Variable(torch.FloatTensor(np.load(directory + "x_train_norm.npy", allow_pickle=True)))
    x_test = Variable(torch.FloatTensor(np.load(directory + "x_test_norm.npy")))

    # Load labels for train and test (numpy array)
    y_train = np.load(directory + "y_train.npy")
    y_test = np.load(directory + "y_test.npy")

    return x_train, y_train, x_test, y_test

def get_cv_dir_names(directory):
    # Get all directory names for cross validation
    dir_names = os.listdir(project_dir + "\\data\\models_prepared\\rnn_formated\\" + directory + "\\")
    dir_names = filter(lambda k: 'CV' in k, dir_names)

    return dir_names

#########################################
# Parse user arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--length_type")
argparser.add_argument("--model_type")
length_type = argparser.parse_args().length_type

if length_type == "dynamic":
    model_directory = "dynamic_length"
elif length_type == "static":
    model_directory = "static_length"
#########################################

#########################################
# Global parameters
dir_names = get_cv_dir_names("fixed_data_length")
cross_statistics = CrossValStatistics()
# List to save all cross validated model validation accuracy
models_accuracy = []
#########################################

#########################################
# Define hyper-parameters
hidden_units = 256
nb_layer = 1
lr = 1e-04  # learning rate
nb_epoch = 100
batch_size = 512
time_step = 60  # rnn time step - here this represents that the RNN would be able to keep in memory the the 60 sensors data
loss_func = nn.CrossEntropyLoss()
#########################################

for folder in dir_names:
    # Check if model with dynamic data length must be called
    if True:
        x_train, y_train, x_test, y_test = get_static_dataset(project_dir + "\\data\\models_prepared\\rnn_formated\\fixed_data_length\\" + folder + "\\")
        input_size = x_train.shape[2]  # rnn input size / nb frames
    else:
        x_train, y_train, x_test, y_test, max_seq_length = get_dynamic_dataset(project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\" + folder + "\\")
        input_size = max_seq_length.max()  # rnn input size / nb frames



    nb_labels = y_train.shape[1]

    # Instantiate our NN
    model_type = argparser.parse_args().model_type
    if model_type == 'GRU':
        print("GRU")
        model = GRU(input_size, hidden_units, nb_layer, nb_labels)
    elif model_type == 'LSTM':
        print("LSTM")
        model = LSTM(input_size, hidden_units, nb_layer, nb_labels)
    elif model_type == 'RNN':
        print("RNN")
        model = RNN(input_size, hidden_units, nb_layer, nb_labels)

    model.cuda()

    # Define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimize all nn parameters

    # Define the loss function
    y_train = torch.tensor(
        np.argmax(y_train, axis=1))  # Get label no one hot encoded for y as Loss of pytorch need it like that
    y_test = torch.tensor(np.argmax(y_test, axis=1))

    # Data Loader for easy mini-batch return in training
    train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=False)

    statistics = Statistics()

    for epoch in range(nb_epoch):
        idx_start = 0
        idx_end = 0

        # scheduler.step()

        for step, x in enumerate(train_loader):  # gives batch data
            model.train()
            idx_end += x.shape[0]

            # reshape x to (batch, time_step, input_size)
            b_x = x.view(-1, time_step, input_size).cuda()
            b_y = Variable(y_train[idx_start:idx_end]).cuda()

            # rnn output
            predict = model(b_x).cuda()

            # cross entropy loss
            output = loss_func(predict, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            training_loss = output.item()

            # Store training loss
            statistics.loss.append(training_loss)

            # backpropagation, compute gradients
            output.backward()

            # Updates parameters
            optimizer.step()

            if step % 13 == 0:
                # Compute training loss and accuracy
                pred_y = torch.max(predict, 1)[1].cpu().data.numpy().squeeze()
                accuracy = sum(pred_y == y_train[idx_start: idx_end].data.numpy()) / y_train[
                                                                                     idx_start: idx_end].data.numpy().size
                print('Epoch: ', epoch, '| train loss: %.4f' % training_loss, '| train accuracy : ', accuracy)
                # Store training accuracy
                statistics.training_accuracy.append(accuracy)

                # Compute testing loss and accuracy
                model.eval()
                x_test_batch = x_test.view(-1, time_step, input_size).cuda()
                test_output = model(x_test_batch) # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()

                accuracy = sum(pred_y == y_test.data.numpy()) / y_test.data.numpy().shape[0]

                print('Test accuracy: %.2f' % accuracy)

                # Store validation accuracy
                statistics.validation_accuracy.append(accuracy)
                # Calculate validation loss
                validation_loss = loss_func(test_output.cuda(), y_test.cuda())
                # Store validation loss
                statistics.validation_loss.append(validation_loss.item())

            idx_start += x.shape[0]

    #statistics.model_structure = "GRU with dropout 0.5, timestep 60, hidden units 256, nb layer 1, nb epoch 100, batch size 512, lr 1e-04"
    statistics.model_structure = ""

    # Store the best validation accuracy in a list (where the validation loss is the lowest)
    best_model_idx = np.argmin(statistics.validation_loss)
    models_accuracy.append(statistics.validation_accuracy[best_model_idx])

    # Store the model history
    cross_statistics.stat_models.append(statistics)

# Store the mean of all cross validation models as the reference accuracy for this model's architecture
cross_statistics.cross_val_accuracy = np.mean(models_accuracy)

path = 'rnn_results\\' + model_directory + '\\'
if not os.path.exists(path):
    os.makedirs(path)
# Save the cross validated model's architecture in a json file
cross_statistics.save(model_type, path)
