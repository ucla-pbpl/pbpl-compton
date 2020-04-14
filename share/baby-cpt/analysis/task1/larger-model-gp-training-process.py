import matplotlib.pyplot as plt
import numpy as np
import re

metrics = ['loss: ','mae: ', 'mse: ', 'val_loss: ', 'val_mae: ', 'val_mse: ']
values = np.zeros((1, len(metrics)))

with open("larger-proper-model-training-process.txt") as fp:
    line = fp.readline()
    while line:
        is_val = line.find("val_loss:")
        if(is_val >=0):
            row = np.zeros((1, len(metrics)))
            for i in range(len(metrics)):
                match = re.search(metrics[i]+'(\d+\.\d+)', line)
                if match:
                    row[0][i] = float(match.group(1))
            #print(row)
            values = np.vstack((values, row))
        line = fp.readline()

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=3)

train_epochs = range(2, len(values))
test_epochs = range(2, len(values))
values = values.T
#loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(train_epochs, values[0][2:], label="loss")
ax1.plot(test_epochs, values[3][2:], label="val_loss")
ax1.set_xlabel("epoch")
ax1.legend()
#mae
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(train_epochs, values[1][2:], label="mae")
ax2.plot(test_epochs, values[4][2:], label="val_mae")
ax2.set_xlabel("epoch")
ax2.legend()
#mse
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(train_epochs, values[2][2:], label="mse")
ax3.plot(test_epochs, values[5][2:], label="val_mse")
ax3.set_xlabel("epoch")
ax3.legend()
plt.show()
                        
