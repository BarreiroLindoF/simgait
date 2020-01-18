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
import math

# Project directory to make this code runnable on any windows system (to be changed on mac)
project_dir = os.path.expanduser(os.path.dirname(os.getcwd()))

# Import markers class to be enable to read data stored in npy files (array of Markers object)
sys.path.append(project_dir + "\\extraction\\models_preprocess\\")
from rnn_preprocess import Markers

torch.autograd.set_detect_anomaly(True)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_units, nb_layer, nb_labels, dropout):
        super(RNN, self).__init__()
        self.num_layers = nb_layer
        self.hidden_dim = hidden_units

        self.rnn = nn.RNN(
            input_size=input_size,  # Size of the input sequence (None if dynamic)
            hidden_size=hidden_units,  # rnn hidden unit
            num_layers=nb_layer,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            nonlinearity='relu',
            dropout=dropout
        )

        self.out = nn.Linear(hidden_units, nb_labels)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        #out, hn = self.rnn(x, h0.detach())
        out, hn = self.rnn(x, h0)

        # Index hidden state of last time step
        # out[:, -1, :] just want last time step hidden states
        out = self.out(out[:, -1, :])

        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_units, nb_layer, nb_labels, dropout):
        super(GRU, self).__init__()
        self.num_layers = nb_layer
        self.hidden_dim = hidden_units
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_dim, dropout=dropout,
                          batch_first=True)
        self.out = nn.Linear(hidden_units, nb_labels)

    def forward(self, x):
        out, hidden = self.gru(x)
        out = self.out(out[:, -1, :])

        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_units, nb_layer, nb_labels, dropout):
        super(LSTM, self).__init__()
        self.num_layers = nb_layer
        self.hidden_dim = hidden_units

        self.lstm = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_units,  # rnn hidden unit
            num_layers=nb_layer,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=dropout
        )

        self.out = nn.Linear(hidden_units, nb_labels)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # 60 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.out(out[:, -1, :])

        return out

#Post-padding/Pre-padding shorten sequence
def get_dynamic_dataset(directory, padding_type="pre_padding"):
    # Load train and validation data (matrix of objects --> Markers)
    X_train_npy = np.load(
       directory + "\\x_train_norm_dynamic.npy",
        allow_pickle=True)
    X_validation_npy = np.load(
        directory + "\\x_test_norm_dynamic.npy",
        allow_pickle=True)

    # Load labels for train and validation (numpy array)
    y_train = np.load(directory + "\\y_train.npy")
    y_validation = np.load(directory + "\\y_test.npy")

    # get lengths of each seq in train and test
    seq_lengths_train = [seq[0].getMarkers().shape[1] for seq in X_train_npy]
    seq_lengths_validation = [seq[0].getMarkers().shape[1] for seq in X_validation_npy]

    if np.max(seq_lengths_train) > np.max(seq_lengths_validation):
        max_seq_length = np.max(seq_lengths_train)
    else:
        max_seq_length = np.max(seq_lengths_validation)

    nb_sensors = X_train_npy[0][0].getMarkers().shape[0]

    # Create a numpy array for each dataset with the shape of the max length sequence with zeros everywhere
    x_train = np.zeros((len(X_train_npy), nb_sensors, max_seq_length))
    x_validation = np.zeros((len(X_validation_npy), nb_sensors, max_seq_length))

    if padding_type == "pre_padding":
        # Fill the zeros tensors with sequences values (each sequence with a length smaller than the max seq lenght will have some 0 padding at the begining)
        for i in range(X_train_npy.shape[0]):
            idx_end_pad = seq_lengths_train[i]
            nb_data_to_add = X_train_npy[i][0].getMarkers().shape[1]
            idx_start_pad = idx_end_pad - nb_data_to_add
            x_train[i, :, idx_start_pad:idx_end_pad] = X_train_npy[i][0].getMarkers()

        for i in range(X_validation_npy.shape[0]):
            idx_end_pad = seq_lengths_validation[i]
            nb_data_to_add = X_validation_npy[i][0].getMarkers().shape[1]
            idx_start_pad = idx_end_pad - nb_data_to_add
            x_validation[i, :, idx_start_pad:idx_end_pad] = X_validation_npy[i][0].getMarkers()
    elif padding_type == "post_padding":
        # Fill the zeros tensors with sequences values (each sequence with a length smaller than the max seq lenght will have some 0 padding at the end)
        for i in range(X_train_npy.shape[0]):
            curr_seq_length = seq_lengths_train[i]
            x_train[i, :, 0:curr_seq_length] = X_train_npy[i][0].getMarkers()

        for i in range(X_validation_npy.shape[0]):
            curr_seq_length = seq_lengths_validation[i]
            x_validation[i, :, 0:curr_seq_length] = X_validation_npy[i][0].getMarkers()

    return x_train, y_train, x_validation, y_validation, max_seq_length


def get_static_dataset(directory):
    # Load data with fixed size (simple array numpy)
    x_train = np.load(directory + "x_train_norm.npy", allow_pickle=True)
    x_validation = np.load(directory + "x_test_norm.npy")

    # Load labels for train and validation (numpy array)
    y_train = np.load(directory + "y_train.npy")
    y_validation = np.load(directory + "y_test.npy")

    return x_train, y_train, x_validation, y_validation

def get_cv_dir_names(directory):
    # Get all directory names for cross validation
    dir_names = os.listdir(project_dir + "\\data\\models_prepared\\rnn_formated\\" + directory + "\\")
    dir_names = filter(lambda k: 'CV' in k, dir_names)

    return dir_names

#########################################
# Parse user arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--length_type", help="Type of the data for the model. Value could be \"dynamic\" or \"static\"")
argparser.add_argument("--model_type", help="Type of the RNN model to instantiate. Value could be \"RNN\", \"LSTM\", \"GRU\"")
argparser.add_argument("--padding_type", help="Type of the padding to do in case of dynamic data length (parameter length_type=dynamic)."
                                              "Value could be \"pre_padding\", \"post_padding\"")
#Length type values can be static or dynamic
length_type = argparser.parse_args().length_type
#Model type values can be RNN, LSTM or GRU
model_type = argparser.parse_args().model_type
#Padding type values can be pre_padding or post_padding
padding_type = argparser.parse_args().padding_type

if length_type == "dynamic":
    model_directory = "dynamic_length"
elif length_type == "static":
    model_directory = "static_length"
#########################################

#########################################
# Global parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dir_names = get_cv_dir_names("fixed_data_length")
cross_statistics = CrossValStatistics()
# List to save all cross validated model validation accuracy
best_models_val_accuracy = []
#List to save all test accuracy (each cross validation data set has a test set)
lst_test_accuracy = []
lst_predictions = []
# Test set variable
x_test = None
y_test = None
# Variables to store mcNemar stats
correct_preds = 0
wrong_preds = 0
#########################################

#########################################
# Define hyper-parameters
hidden_units = 200
dropout = 0.3
nb_layer = 5
lr = 1e-04  # learning rate
nb_epoch = 30
batch_size = 128
time_step = 60  # rnn time step - here this represents that the RNN would be able to keep in memory the the 60 sensors data
loss_func = nn.CrossEntropyLoss()
#########################################

#########################################
# Train and save model
for folder in dir_names:
    # Check if model with dynamic data length must be called
    if length_type == 'static':
        # Get data in a static data length way
        x_train, y_train, x_validation, y_validation = get_static_dataset(project_dir + "\\data\\models_prepared\\rnn_formated\\fixed_data_length\\" + folder + "\\")
        input_size = int(x_train.shape[2])  # rnn input size / nb frames
    elif length_type == 'dynamic':
        # Make padding for dynamic data length usage
        x_train, y_train, x_validation, y_validation, max_seq_length = get_dynamic_dataset(project_dir + "\\data\\models_prepared\\rnn_formated\\dynamic_data_length\\" + folder + "\\", padding_type)
        input_size = int(max_seq_length.max())  # rnn input size / nb frames

    nb_labels = y_train.shape[1]

    # Define validation and test data set as 50% of the total test data set of the current cross validation data
    length_validation = math.trunc(x_validation.shape[0] / 2)
    length_test = x_validation.shape[0] - length_validation

    # Define test set for the current cross validation data
    x_test = x_validation[:length_test, :, :]
    y_test = y_validation[:length_test, :]

    # Define validation set for the current cross validation data
    x_validation = x_validation[length_validation:, :, :]
    y_validation = y_validation[length_validation:, :]

    # Get label no one hot encoded for y as Loss of Pytorch need it like that
    y_train = np.argmax(y_train, axis=1)
    y_validation = np.argmax(y_validation, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print("\nTransfering data to GPU or CPU depending on PC hardware available")
    x_train = torch.from_numpy(x_train).to(device, dtype=torch.float)
    y_train = torch.from_numpy(y_train).to(device, dtype=torch.long)
    x_validation = torch.from_numpy(x_validation).to(device, dtype=torch.float)
    y_validation = torch.from_numpy(y_validation).to(device, dtype=torch.long)
    x_test = torch.from_numpy(x_test).to(device, dtype=torch.float)
    y_test = torch.from_numpy(y_test).to(device, dtype=torch.long)

    # Instantiate our NN
    if model_type == 'GRU':
        print("Instantiate GRU")
        model = GRU(input_size, hidden_units, nb_layer, nb_labels, dropout)
    elif model_type == 'LSTM':
        print("Instantiate LSTM")
        model = LSTM(input_size, hidden_units, nb_layer, nb_labels, dropout)
    elif model_type == 'RNN':
        print("Instantiate RNN")
        model = RNN(input_size, hidden_units, nb_layer, nb_labels, dropout)

    model.to(device)

    # Define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimize all nn parameters

    # Data Loader for easy mini-batch return in training
    train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=False)

    statistics = Statistics()

    for epoch in range(nb_epoch):
        idx_start = 0
        idx_end = 0

        for step, x in enumerate(train_loader):  # gives batch data
            model.train()
            idx_end += x.shape[0]

            # reshape x to (batch, time_step, input_size)
            b_x = x.view(-1, time_step, input_size).to(device)
            b_y = Variable(y_train[idx_start:idx_end]).to(device)

            # rnn output
            predict = model(b_x)

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
                train_accuracy = sum(pred_y == y_train[idx_start: idx_end].cpu().data.numpy()) / y_train[
                                                                                     idx_start: idx_end].cpu().data.numpy().size
                # Store training accuracy
                statistics.training_accuracy.append(train_accuracy)

                # Compute validation loss and accuracy
                model.eval()
                x_validation_batch = x_validation.view(-1, time_step, input_size).to(device)
                validation_output = model(x_validation_batch) # (samples, time_step, input_size)
                pred_y = torch.max(validation_output, 1)[1].cpu().data.numpy().squeeze()

                accuracy = sum(pred_y == y_validation.cpu().data.numpy()) / y_validation.cpu().data.numpy().shape[0]


                # Store validation accuracy
                statistics.validation_accuracy.append(accuracy)
                # Calculate validation loss
                validation_loss = loss_func(validation_output.to(device), y_validation.to(device))
                # Store validation loss
                statistics.validation_loss.append(validation_loss.item())

                print('Epoch: ', epoch, '| train loss: %.4f' % training_loss, '| train accuracy : ', train_accuracy)
                print('Validation loss: %.4f' % validation_loss, '| Validation accuracy : ', accuracy, end="\r")

            idx_start += x.shape[0]

    #statistics.model_structure = "GRU with dropout 0.5, timestep 60, hidden units 256, nb layer 1, nb epoch 100, batch size 512, lr 1e-04"
    statistics.model_structure = '' + str(model_type) + ' with ' + str(time_step) + ' timesteps, ' + str(hidden_units) + ' hidden units, ' + \
                                 str(nb_layer) + ' layer, ' + str(nb_epoch) + ' epochs, ' + ' batch size of ' + str(batch_size) + ', learning rate of ' + str(lr) + \
                                 ', with ' + length_type + ' length of data and ' + padding_type + ' type.'

    # Store the best validation accuracy in a list (where the validation loss is the lowest)
    best_model_idx = np.argmin(statistics.validation_loss)
    best_models_val_accuracy.append(statistics.validation_accuracy[best_model_idx])

    # Compute accuracy on test set
    model.eval()
    test_output = model(x_test)
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()

    # Store test accuracy
    test_accuracy = sum(pred_y == y_test.cpu().data.numpy()) / y_test.cpu().data.numpy().shape[0]
    lst_test_accuracy.append(test_accuracy)
    statistics.test_accuracy = test_accuracy

    # Store correct and wrong predictions on current CV for McNemar contingency table
    cross_statistics.predictionResults.append((pred_y == y_test.cpu().data.numpy()).tolist())

    # Store the model history
    cross_statistics.stat_models.append(statistics)

# Store the mean accuracy over all cross validated models to have the general validation accuracy on this current model
cross_statistics.validation_mean_accuracy = np.mean(best_models_val_accuracy)

# Store the mean accuracy over all cross validated models to have the general testing accuracy on this current model
cross_statistics.test_mean_accuracy = np.mean(lst_test_accuracy)
print('\nTest accuracy mean : ', np.mean(lst_test_accuracy))

path = 'rnn_results\\' + model_directory + '\\'
if not os.path.exists(path):
    os.makedirs(path)
# Save the cross validated model's architecture in a json file
cross_statistics.save(model_type, path)
