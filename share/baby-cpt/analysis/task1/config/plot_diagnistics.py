import csv
import numpy as np
import matplotlib.pyplot as plt

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

filename = "../grand_mse_full.csv"
def string_to_float(str_list):
    res = []
    for s in str_list:
        try:
            i = float(s)
            res.append(i)
            #print (i)
        except ValueError:
            #Handle the exception
            #print (ValueError)
            pass
    return np.array(res)

def read_column(column_name, numerical=False):
    read_in = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter=',')
        row1 = next(reader) 
        index = row1.index(column_name)
        for row in reader:
            read_in.append(row[index])
    if numerical:
        return string_to_float(np.transpose(np.array(read_in)))
    else:
        return read_in

mse = read_column("mse", True)
learning_rate = read_column("learning rate")

lr_sm = []
lr_big = []

for i in range(len(mse)):
    if learning_rate[i]=="0.001":
        lr_sm.append(mse[i])
    else:
        lr_big.append(mse[i])

fig = plt.figure(constrained_layout=True, figsize=(3+3/8, 4+4/8), dpi=600)
gs = fig.add_gridspec(nrows=2, ncols=1)

ax1 = fig.add_subplot(gs[0, 0])
ax1.hist([lr_sm, lr_big], bins=20, label=["lr=0.001", "lr=0.01"], rwidth=0.9)
ax1.set_xlabel("mse on testing set")
ax1.set_ylabel("count")
ax1.legend()
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
layer_count = read_column("layer count")

layer3 = []
layer4 = []

for i in range(len(mse)):
    if layer_count[i]=="4":
        layer4.append(mse[i])
    else:
        layer3.append(mse[i])

ax2 = fig.add_subplot(gs[1, 0])
ax2.hist([layer3, layer4], bins=20, label=["3 layers", "4 layers"], rwidth=0.9)
ax2.set_xlabel("mse on testing set")
ax2.set_ylabel("count")
ax2.legend()
ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig("diag.png")



