from statistic_saver import Statistics, CrossValStatistics
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import tikzplotlib


# Project directory to make this code runnable on any windows system (to be changed on mac)
project_dir = os.path.expanduser(os.path.dirname(os.getcwd()))

cross_statistics = CrossValStatistics()
##################################################
# Make the comparison between all 3 best RNN types
##################################################
# Best RNN directories
best_rnn_dir = join(project_dir, "models", "rnn_results", "static_length")
best_lstm_dir = join(project_dir, "models", "rnn_results", "dynamic_length")
best_gru_dir = join(project_dir, "models", "rnn_results", "dynamic_length")

# Load three best recurrent models history
cross_statistics_rnn = cross_statistics.load(filename="RNN", folder=best_rnn_dir)
cross_statistics_lstm = cross_statistics.load(filename="LSTM", folder=best_lstm_dir)
cross_statistics_gru = cross_statistics.load(filename="GRU", folder=best_gru_dir)

# Get number of epochs on the validation_accuracy length for example. (could be another list)
nb_epochs = len(cross_statistics_rnn.stat_models[0]['validation_accuracy'])

# Create empty array to store validation accuracy mean over all cross validation set for each best model
rnn_mean_val_accuracy = np.zeros(nb_epochs)
lstm_mean_val_accuracy = np.zeros(nb_epochs)
gru_mean_val_accuracy = np.zeros(nb_epochs)

# Create empty array to store training accuracy mean for the best recurrent model
gru_mean_train_accuracy = np.zeros(nb_epochs)

# Get the number of cross validation set
nb_cv = len(cross_statistics_rnn.stat_models)

# Loop through all cross validated model history and get validation accuracy history for all and training accuracy history just for the best one
for i in range(nb_cv):
    rnn_mean_val_accuracy += np.asarray(cross_statistics_rnn.stat_models[i]['validation_accuracy'])
    lstm_mean_val_accuracy += np.asarray(cross_statistics_lstm.stat_models[i]['validation_accuracy'])
    gru_mean_val_accuracy += np.asarray(cross_statistics_gru.stat_models[i]['validation_accuracy'])

    gru_mean_train_accuracy += np.asarray(cross_statistics_lstm.stat_models[i]['training_accuracy'])

# Divide the histories by the number of cross validated set to get the mean
rnn_mean_val_accuracy /= nb_cv
lstm_mean_val_accuracy /= nb_cv
gru_mean_val_accuracy /= nb_cv
gru_mean_train_accuracy /= nb_cv

# Plot the three validation model accuracy with a convolution to make them smoother
y = np.convolve(rnn_mean_val_accuracy, [0.3, 0.3, 0.3])[:-2]
plt.plot(y, color="red")
y = np.convolve(lstm_mean_val_accuracy, [0.3, 0.3, 0.3])[:-2]
plt.plot(y, color="blue")
y = np.convolve(gru_mean_val_accuracy, [0.3, 0.3, 0.3])[:-2]
plt.plot(y, color="green")
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.legend(('RNN', 'LSTM', 'GRU'))
# Create the tickz file to include it easier in the report
tikzplotlib.save('validationAllModels.tex')
plt.show()

# Plot the best model training and validation accuracy with a convolution to make it smoother
y = np.convolve(gru_mean_val_accuracy, [0.3, 0.3, 0.3])[:-2]
plt.plot(y, color="blue")
y = np.convolve(gru_mean_train_accuracy, [0.3, 0.3, 0.3])[:-2]
plt.plot(y, color="green")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(('Validation', 'Train'))
# Create the tickz file to include it easier in the report
tikzplotlib.save('validationTrainBest.tex')
plt.show()

