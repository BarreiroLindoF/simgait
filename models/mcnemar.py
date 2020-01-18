from statistic_saver import Statistics, CrossValStatistics
import os

# Project directory to make this code runnable on any windows system (to be changed on mac)
project_dir = os.path.expanduser(os.path.dirname(os.getcwd()))

# RNN directory
rnn_dir = "rnn_results"

# CNN directory
cnn_dir = "cnn_results"

cross_statistics = CrossValStatistics()

cross_statistics = cross_statistics.load(filename="RNN", folder=os.path.join(project_dir, "models", rnn_dir, "dynamic_length"))

