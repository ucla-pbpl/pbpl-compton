import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

x_max = 245#= np.max(points[:, 0])+1
#print(x_max)
x_bins = int(x_max/2)
y_max = 156 #= np.max(points[:, 1])+1
#print(y_max)
y_bins = 128
    
def get_image(points, energy):
    try:
        zi, xi, yi = np.histogram2d(points[:,0].flatten(), points[:,1].flatten(), bins=(x_bins, y_bins), weights=energy, normed=False, range=[[0, x_max], [-y_max, y_max]])
    except IndexError:
        return None
    return zi

def main():
    train_dir = ['train-right-y/']
    test_dir = 'test-right-y/'
    data_file = 'set0-right-y-mixed-big.npz'
    #e_threshold = 50

    train_files = []
    for dirname in train_dir:
        train_files.extend(glob.glob(dirname+'*-col-*.h5'))
    test_files = glob.glob(test_dir+'*-col-*.h5')
    train_data = np.empty([1, x_bins, y_bins])
    train_labels = [np.linspace(0, 31, 50*50)]
    test_data = np.empty([1, x_bins, y_bins])
    test_labels = [np.linspace(0, 31, 50*50)]

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
       
        
        b_out = get_image(points, energy)
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
       
        
        b_out = get_image(points, energy)
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
