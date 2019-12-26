import matplotlib.pyplot as plt
import numpy as np
from statistic_saver import Statistics

cnn_3layer = Statistics.load('cnn', 'cnns')
cnn_reg_loss = Statistics.load('cnn_reg_loss', 'cnns')
cnn_2layer = Statistics.load('cnn_2layer', 'cnns')
"""
rnn = Statistics.load('rnn', 'rnn_stats')
lstm = Statistics.load('lstm', 'rnn_stats')
gru = Statistics.load('gru', 'rnn_stats')


x = np.arange(0,100)
y = rnn.validation_accuracy
plt.plot(x, y, color="red")
y = lstm.validation_accuracy
plt.plot(x, y, color="blue")
# y = cnn_reg_loss.validation_accuracy
# plt.plot(x, y)

y = gru.validation_accuracy
plt.plot(x, y, color="green")
# plt.savefig("test.svg", format="svg")
"""
x = np.arange(0,100)
y = np.convolve(cnn_3layer.validation_accuracy, [0.3,0.3,0.3])[:-2]
plt.plot(x, y, color="red")

y = np.convolve(cnn_reg_loss.validation_accuracy, [0.3,0.3,0.3])[:-2]
plt.plot(x, y, color="green")
y = np.convolve(cnn_2layer.validation_accuracy, [0.3,0.3,0.3])[:-2]
plt.plot(x, y, color="blue")

plt.show()

"""
#Make the mean of cnn and mean of rnn values on validation set to have smoother graph
mean_cnn = (np.array(cnn_3layer.validation_accuracy) + np.array(cnn_reg.validation_accuracy) + np.array(cnn_reg_loss.validation_accuracy) + np.array(cnn_2layer.validation_accuracy)) / 4
mean_rnn = (np.array(rnn.validation_accuracy) + np.array(lstm.validation_accuracy) + np.array(gru.validation_accuracy)) / 3

print(mean_cnn.shape)
print(mean_rnn.shape)

x = np.arange(0,100)
y = mean_cnn
plt.plot(x,y, color="green")

y = mean_rnn
plt.plot(x,y, color="red")
plt.show()
"""