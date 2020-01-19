from statistic_saver import Statistics, CrossValStatistics
import os
from os.path import join
import numpy as np

# Project directory to make this code runnable on any windows system (to be changed on mac)
project_dir = os.path.expanduser(os.path.dirname(os.getcwd()))

cross_statistics = CrossValStatistics()


def get_frequencies(model_1, model_2) :
    # Case wrong by the model 1 and correct by the model 2
    wrong_1_correct_2 = np.count_nonzero(np.logical_and(model_1 == False, model_2))
    # Case correct by the model 1 and wrong by the model 2
    correct_1_wrong_2 = np.count_nonzero(np.logical_and(model_1, model_2 == False))
    # Case wrong by the model 1 and wrong by the model 2
    wong_1_wrong_2 = np.count_nonzero(np.logical_and(model_1 == False, model_2 == False))
    # Case correct by the model 1 and correct by the model 2
    correct_1_correct_2 = np.count_nonzero(np.logical_and(model_1, model_2))

    theoritical_frequencies = (wrong_1_correct_2 + correct_1_wrong_2) / 2

    p_value = (((wrong_1_correct_2 - theoritical_frequencies) ** 2) / theoritical_frequencies) + (((correct_1_wrong_2 - theoritical_frequencies) ** 2) / theoritical_frequencies)

    return  correct_1_correct_2, correct_1_wrong_2, wrong_1_correct_2,wong_1_wrong_2, p_value

# Make the comparison between all 3 best RNN types
# Best RNN directories
best_rnn_dir = join(project_dir, "models", "rnn_results", "static_length")
best_rnn2_dir = join(project_dir, "models", "rnn_results", "dynamic_length")
best_gru_dir = join(project_dir, "models", "rnn_results", "static_length")

# Load three best recurrent models history
cross_statistics_rnn = cross_statistics.load(filename="RNN", folder=best_rnn_dir)
cross_statistics_rnn2 = cross_statistics.load(filename="RNN", folder=best_rnn2_dir)
cross_statistics_gru = cross_statistics.load(filename="GRU", folder=best_gru_dir)

# Get correct and wrong predictions for each model history
predictions_rnn = np.concatenate(cross_statistics_rnn.predictionResults)
predictions_lstm = np.concatenate(cross_statistics_rnn2.predictionResults)
predictions_gru = np.concatenate(cross_statistics_gru.predictionResults)


# Observed frequencies (m1_c_m2_c = model 1 nb of correct and correct in model 2 too)
m1_c_m2_c, m1_c_m2_w, m1_w_m2_c, m1_w_m2_w, p_value = get_frequencies(predictions_rnn, predictions_lstm)
print('McNemar significiance between RNN and LSTM : ', p_value)
m1_c_m2_c, m1_c_m2_w, m1_w_m2_c, m1_w_m2_w, p_value = get_frequencies(predictions_rnn, predictions_gru)
print('McNemar significiance between RNN and GRU : ', p_value)
m1_c_m2_c, m1_c_m2_w, m1_w_m2_c, m1_w_m2_w, p_value = get_frequencies(predictions_lstm, predictions_gru)
print('McNemar significiance between LSTM and GRU : ', p_value)

# Make the comparison between all best CNN types
# Best CNN directories
########################################
# Just copy code between line 34-51 and adapt for cnn (like directories path etc)
########################################
