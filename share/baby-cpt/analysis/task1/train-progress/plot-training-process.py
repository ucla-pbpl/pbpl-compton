import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib as mpl

def setup_plot():
    mpl.rc('font', size=8.0, family='sans-serif')
    mpl.rc('mathtext', fontset='cm')
    mpl.rc('pdf', use14corefonts=True)
    mpl.rc('lines', linewidth=0.8)
    mpl.rc('xtick.major', size=4)
    mpl.rc('ytick.major', size=4)
    mpl.rc('xtick', direction='in', top=True)
    mpl.rc('xtick.minor', size=2)
    mpl.rc('ytick.minor', size=2)
    mpl.rc('ytick', direction='in', right=True)
    mpl.rc('legend', fontsize=7.0)
    mpl.rc('legend', labelspacing=0.25)
    mpl.rc('legend', frameon=False)
    mpl.rc('path', simplify=False)
    mpl.rc('axes', unicode_minus=False, linewidth=0.8)
    mpl.rc('figure.subplot', right=0.97, top=0.97, bottom=0.15, left=0.13)
    mpl.rc('axes', titlepad=-10)
    mpl.rc('axes', prop_cycle=mpl.cycler(
        'color', [
            '#0083b8', '#e66400', '#93a661', '#ebc944', '#da1884', '#7e48bd']))
    # girl scouts green is 0x00ae58
    mpl.rc('figure', max_open_warning=False)

setup_plot()
metrics = ['loss: ','mae: ', 'mse: ', 'val_loss: ', 'val_mae: ', 'val_mse: ']
values = np.zeros((1, len(metrics)))

#with open("../Thalheimer/log.txt") as fp:
with open("../../nonstripe4500/Thabit/log.txt") as fp:
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

fig = plt.figure(constrained_layout=True, figsize=(3+3/8, 3+3/8), dpi=600)
gs = fig.add_gridspec(nrows=1, ncols=1)

train_epochs = range(2, len(values))
test_epochs = range(2, len(values))
values = values.T
#loss
#ax1 = fig.add_subplot(gs[0, 0])
#ax1.plot(train_epochs, values[0][2:], label="loss")
#ax1.plot(test_epochs, values[3][2:], label="val_loss")
#ax1.set_xlabel("epoch")
#ax1.legend()
#mae
#ax2 = fig.add_subplot(gs[0, 1])
#ax2.plot(train_epochs, values[1][2:], label="mae")
#ax2.plot(test_epochs, values[4][2:], label="val_mae")
#ax2.set_xlabel("epoch")
#ax2.legend()
#mse
ax3 = fig.add_subplot(gs[0, 0])
ax3.plot(train_epochs, values[2][2:], label="mse")
ax3.plot(test_epochs, values[5][2:], label="val_mse")
ax3.set_xlabel("epoch")
ax3.legend()
plt.savefig("train-progress_thabit.png")
                        
