import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from model import Cnn
from sklearn.utils import class_weight
from loss import FocalLoss
from statistic_saver import Statistics, CrossValStatistics
import os
import argparse

#########################################
# Parse user argument and set parameters
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_type")
arg_model_type = argparser.parse_args().model_type

if arg_model_type == "100M":
    model_directory = "Width100MarkersOnly"
    # If there are only markers (20 markers x 3 axis = 60), otherwise there are 84 features (24 angles + 20 x 3 markers)
    features_size = 60
    flatten_size = 960
elif arg_model_type == "200MA":
    model_directory = "Width200MarkersAngles"
    # If there are only markers (20 markers x 3 axis = 60), otherwise there are 84 features (24 angles + 20 x 3 markers)
    features_size = 84
    flatten_size = 1984
elif arg_model_type == "200M":
    model_directory = "Width200MarkersOnly"
    # If there are only markers (20 markers x 3 axis = 60), otherwise there are 84 features (24 angles + 20 x 3 markers)
    features_size = 60
    flatten_size = 1984
#########################################

#########################################
# Global parameters
# Set seed to have reproductible results
torch.manual_seed(0)

# Project directory to make this code runnable on any windows system (to be changed on mac)

project_dir = os.path.expanduser(os.path.dirname(os.getcwd()))
# List to save all cross validated model validation accuracy
models_accuracy = []

# Get all directory names for cross validation
dir_names = os.listdir(project_dir + "\\data\\models_prepared\\cnn_formated\\" + model_directory + "\\")
dir_names = filter(lambda k: 'CV' in k, dir_names)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cross_statistics = CrossValStatistics()
#########################################

#########################################
# Hyper parameters
n_epochs = 100
batch_size = 32
#########################################

# weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
# weights = np.append(weights, 0)
# weights[4] = 20
# print(weights)

# weights = torch.from_numpy(weights).to(device, dtype=torch.float)


# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)

print("Intializating model and parameters")

loss = nn.CrossEntropyLoss()
cnn = nn.Sequential(nn.Conv1d(features_size, 256, 5),
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
                    nn.Linear(flatten_size, 128),
                    # nn.Dropout(0.5),
                    #nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 6))

cnn.to(device)
init_weights = cnn.state_dict()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

print("Starting training")

# First loop to load each cross validated dataset
for folder in dir_names:
    # Initialize the statistics class
    statistics = Statistics()
    # Initialize the weights for this model
    cnn.load_state_dict(init_weights)

    print("Loading data")
    X_train = np.load(
        project_dir + "\\data\\models_prepared\\cnn_formated\\" + model_directory + "\\" + folder + "\\1d_X_train.npy")
    y_train = np.load(
        project_dir + "\\data\\models_prepared\\cnn_formated\\" + model_directory + "\\" + folder + "\\1d_y_train.npy")
    y_train = np.argmax(y_train, axis=1)
    X_test = np.load(project_dir + "\\data\\models_prepared\\cnn_formated\\" + model_directory + "\\" + folder + "\\1d_X_test.npy")
    y_test = np.load(project_dir + "\\data\\models_prepared\\cnn_formated\\" + model_directory + "\\" + folder + "\\1d_y_test.npy")
    y_test = np.argmax(y_test, axis=1)
    print(X_train.shape)
    print("Transfering data to GPU")
    X_train = torch.from_numpy(X_train).to(device, dtype=torch.float)
    y_train = torch.from_numpy(y_train).to(device, dtype=torch.long)
    X_test = torch.from_numpy(X_test).to(device, dtype=torch.float)
    y_test = torch.from_numpy(y_test).to(device, dtype=torch.long)

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

            training_loss = output.item()
            # Store training loss
            statistics.loss.append(training_loss)

            # Calculate training accuracy
            training_predictions = F.softmax(predict)
            _, predicted = torch.max(training_predictions.data, 1)
            correct = (predicted == batch_y).sum()
            # Store training accuracy
            statistics.training_accuracy.append(correct.cpu().numpy() / y_train.shape[0])

        cnn.eval()
        with torch.no_grad():
            # Calculate validation accuracy
            predict_val = cnn.forward(X_test)
            validation_predictions = F.softmax(predict_val)
            _, predicted = torch.max(validation_predictions.data, 1)
            correct = (predicted == y_test).sum()
            # Store validation accuracy
            statistics.validation_accuracy.append(correct.cpu().numpy() / y_test.shape[0])
            # Calculate validation loss
            validation_loss = loss(predict_val, y_test)
            # Store validation loss
            statistics.validation_loss.append(validation_loss.item())

        print('training_loss:', training_loss, "validation accuracy", correct.cpu().numpy() / y_test.shape[0], end="\r")
        cnn.train()

    # Store the best validation accuracy in a list (where the validation loss is the lowest)
    best_model_idx = np.argmin(statistics.validation_loss)
    models_accuracy.append(statistics.validation_accuracy[best_model_idx])

    # Store the model history
    cross_statistics.stat_models.append(statistics)

    print("\nAccuracy with minimum loss for each Cross validation set : ", models_accuracy)

# Store the mean of all cross validation models as the reference accuracy for this model's architecture
cross_statistics.cross_val_accuracy = np.mean(models_accuracy)

path = 'cnn_results\\' + model_directory + '\\'
if not os.path.exists(path):
    os.makedirs(path)
# Save the cross validated model's architecture in a json file
cross_statistics.save('cnn_2layer', path)


