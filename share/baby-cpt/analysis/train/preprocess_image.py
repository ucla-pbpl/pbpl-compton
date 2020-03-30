import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
import toml
    
def get_image(points, energy, x_bins, x_max, y_bins, y_max):
    try:
        zi, xi, yi = np.histogram2d(points[:,0].flatten(), points[:,1].flatten(), 
        bins=(x_bins, y_bins), weights=energy, normed=False, 
        range=[[0, x_max], [-y_max, y_max]])
    except IndexError:
        return None
    return zi

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess raw data into training and testing sets.')
    parser.add_argument("--train_dir", required=True, nargs="+", 
        help="set where the training set comes from.")
    parser.add_argument("--test_dir", required=True, nargs="+", 
        help="set where the testing set comes from.")
    parser.add_argument("--data_file", "-O", required=True,
        help="set name of the file to save the training and testing sets in.")
    parser.add_argument("--config", required=True,
        help="the config toml file")
    
    args = parser.parse_args()

    conf = toml.load(args.config)
    train_dir = args.train_dir
    test_dir = args.test_dir
    data_file = args.data_file
    #e_threshold = 50
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    x_max = int(conf['Simulation']['XMax'])
    #print(x_max)
    x_bins = int(conf['Simulation']['XBins'])
    y_max = int(conf['Simulation']['YMax'])
    #print(y_max)
    y_bins = int(conf['Simulation']['YBins'])
    
    train_files = []
    for dirname in train_dir:
        train_files.extend(glob.glob(dirname+'/*-col-*.h5'))
    print(train_files)
    test_files = []
    for dirname in test_dir:
        test_files.extend(glob.glob(dirname+'/*-col-*.h5'))
    train_data = np.empty([1, x_bins, y_bins])
    train_labels = np.array([np.linspace(0, 31, l_e_bins*l_y_bins)])
    print(train_labels.shape)
    test_data = np.empty([1, x_bins, y_bins])
    test_labels = np.array([np.linspace(0, 31, l_e_bins*l_y_bins)])

    for tr_f in train_files:
        energy_label=[]
        try:
            #base = os.path.basename(tr_f)
            name = os.path.splitext(tr_f)[0]
            with np.load(name+'.npz') as data:
                energy_label = data['histo'].flatten()
                print(energy_label)
        except IOError:
            continue
        print('reading file '+tr_f)
        f = h5py.File(tr_f, 'r')
        points = np.array(f['position'])
        energy = np.array(f['edep'])
        #print(points.shape)
       
        
        b_out = get_image(points, energy, x_bins, x_max, y_bins, y_max)
        #print(b_out.shape)
        if not (b_out is None):
            train_data = np.append(train_data, [b_out], axis=0)
            train_labels = np.append(train_labels, [energy_label], axis = 0)
        
        #fig, ax = plt.subplots()
        #ax.pcolormesh(xi, yi, np.transpose(zi))
        #plt.show()    
        #print(train_data.shape)

    for ts_f in test_files:
        energy_label=[]
        try:
            name = os.path.splitext(ts_f)[0]
            with np.load(name+'.npz') as data:
                energy_label = data['histo'].flatten()
        except IOError:
            continue
        print('reading file '+ts_f)
        f = h5py.File(ts_f, 'r')
        points = np.array(f['position'])
        energy = np.array(f['edep'])
        #print(points.shape)
       
        
        b_out = get_image(points, energy, x_bins, x_max, y_bins, y_max)
        #print(b_out.shape)
        if not (b_out is None):
            test_data = np.append(test_data, [b_out], axis=0)
            test_labels = np.append(test_labels, [energy_label], axis = 0)
        
        #fig, ax = plt.subplots()
        #ax.pcolormesh(xi, yi, np.transpose(zi))
        #plt.show()    
        #print(train_data.shape)
         
    print(train_data.shape)
    train_data = train_data[1:].astype(float)
    train_labels = train_labels[1:].astype(float)
    test_data = test_data[1:].astype(float)
    test_labels = test_labels[1:].astype(float)
    np.savez(data_file, train_data=train_data, train_labels=train_labels, 
                test_data=test_data, test_labels=test_labels)
    print('done savine '+data_file)

if __name__ == "__main__":
   main()
