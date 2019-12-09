import matplotlib.pyplot as plt
import numpy as np
from statistic_saver import Statistics
"""
cnn_3layer = Statistics.load('cnn', 'cnns')
cnn_reg = Statistics.load('cnn_reg', 'cnns')
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
plt.show()
# plt.savefig("test.svg", format="svg")
