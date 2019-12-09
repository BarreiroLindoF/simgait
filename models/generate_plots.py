import matplotlib.pyplot as plt
import numpy as np
from statistic_saver import Statistics

cnn_3layer = Statistics.load('cnn', 'cnns')
cnn_reg = Statistics.load('cnn_reg', 'cnns')
cnn_reg_loss = Statistics.load('cnn_reg_loss', 'cnns')
cnn_2layer = Statistics.load('cnn_2layer', 'cnns')

x = np.arange(0,100)
y = cnn_3layer.validation_accuracy
plt.plot(x, y)
y = cnn_reg.validation_accuracy
plt.plot(x, y)
# y = cnn_reg_loss.validation_accuracy
# plt.plot(x, y)

y = cnn_2layer.validation_accuracy
plt.plot(x, y)
plt.show()
# plt.savefig("test.svg", format="svg")
