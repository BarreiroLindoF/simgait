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
import math

#########################################
# Parse user argument and set parameters
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_type", help="The type of the model to be called. Can be 100M, 200MA or 200M."
                                            "M is markers only, MA is Markers and Angles. 100 and 200 is the size")
argparser.add_argument("--nb_epochs", help="Number of epochs to do. Must be an integer")
nb_epochs = argparser.parse_args().nb_epochs
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

# Variables to store mcNemar stats
correct_preds = 0
wrong_preds = 0
#List to save all test accuracy (each cross validation data set has a test set)
lst_test_accuracy = []

cross_statistics = CrossValStatistics()
#########################################

#########################################
# Hyper parameters
n_epochs = nb_epochs
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
    x_train = np.load(
        project_dir + "\\data\\models_prepared\\cnn_formated\\" + model_directory + "\\" + folder + "\\1d_X_train.npy")
    y_train = np.load(
        project_dir + "\\data\\models_prepared\\cnn_formated\\" + model_directory + "\\" + folder + "\\1d_y_train.npy")
    y_train = np.argmax(y_train, axis=1)
    x_validation = np.load(project_dir + "\\data\\models_prepared\\cnn_formated\\" + model_directory + "\\" + folder + "\\1d_X_test.npy")
    y_validation = np.load(project_dir + "\\data\\models_prepared\\cnn_formated\\" + model_directory + "\\" + folder + "\\1d_y_test.npy")
    y_validation = np.argmax(y_validation, axis=1)

    # Define validation and test data set as 50% of the total test data set of the current cross validation data
    length_validation = math.trunc(x_validation.shape[0] / 2)
    length_test = x_validation.shape[0] - length_validation

    # Define test set for the current cross validation data
    x_test = x_validation[:length_test, :, :]
    y_test = y_validation[:length_test]

    # Define validation set for the current cross validation data
    x_validation = x_validation[length_validation:, :, :]
    y_validation = y_validation[length_validation:]

    print("Transfering data to GPU or CPU depending on PC hardware available")
    x_train = torch.from_numpy(x_train).to(device, dtype=torch.float)
    y_train = torch.from_numpy(y_train).to(device, dtype=torch.long)
    x_validation = torch.from_numpy(x_validation).to(device, dtype=torch.float)
    y_validation = torch.from_numpy(y_validation).to(device, dtype=torch.long)
    x_test = torch.from_numpy(x_test).to(device, dtype=torch.float)
    y_test = torch.from_numpy(y_test).to(device, dtype=torch.long)

    #loss = FocalLoss()
    for epoch in range(n_epochs):

        permutation = torch.randperm(x_train.shape[0])
        for i in range(0, x_train.shape[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
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
            predict_val = cnn.forward(x_validation)
            validation_predictions = F.softmax(predict_val)
            _, predicted = torch.max(validation_predictions.data, 1)
            correct = (predicted == y_validation).sum()
            # Store validation accuracy
            statistics.validation_accuracy.append(correct.cpu().numpy() / y_validation.shape[0])
            # Calculate validation loss
            validation_loss = loss(predict_val, y_validation)
            # Store validation loss
            statistics.validation_loss.append(validation_loss.item())

        print('training_loss:', training_loss, "validation accuracy", correct.cpu().numpy() / y_validation.shape[0], end="\r")
        cnn.train()

    # Store the best validation accuracy in a list (where the validation loss is the lowest)
    best_model_idx = np.argmin(statistics.validation_loss)
    models_accuracy.append(statistics.validation_accuracy[best_model_idx])

    # Compute accuracy on test set
    cnn.eval()
    test_output = cnn(x_test)
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()

    # Store test accuracy
    test_accuracy = sum(pred_y == y_test.cpu().data.numpy()) / y_test.cpu().data.numpy().shape[0]
    lst_test_accuracy.append(test_accuracy)
    statistics.test_accuracy = test_accuracy

    # Store correct and wrong predictions on current CV for McNemar contingency table
    cross_statistics.predictionResults.append((pred_y == y_test.cpu().data.numpy()).tolist())

    # Store the model history
    cross_statistics.stat_models.append(statistics)

    print("\nAccuracy with minimum loss for each Cross validation set : ", models_accuracy)

# Store the mean of all cross validation models as the reference accuracy for this model's architecture
cross_statistics.validation_mean_accuracy = np.mean(models_accuracy)

# Store the mean accuracy over all cross validated models to have the general testing accuracy on this current model
cross_statistics.test_mean_accuracy = np.mean(lst_test_accuracy)
print('Test accuracy mean : ', np.mean(lst_test_accuracy))

path = 'cnn_results\\' + model_directory + '\\'
if not os.path.exists(path):
    os.makedirs(path)
# Save the cross validated model's architecture in a json file
cross_statistics.save('cnn_2layer', path)


