import numpy as np
import toml
import matplotlib.pyplot as plt
import argparse
import h5py

def main():
    parser = argparse.ArgumentParser(
        description='Train network based on data file')
    parser.add_argument("--data_file_name", required=True, nargs="+",
        help="set where the training data comes from.")
    parser.add_argument("--config", required=True, 
        help="set dimensions of input and output data")
    args = parser.parse_args()
    
    conf = toml.load(args.config)
    x_max = int(conf['Simulation']['XMax'])
    x_bins = int(conf['Simulation']['XBins'])
    y_max = int(conf['Simulation']['YMax'])
    y_bins = int(conf['Simulation']['YBins'])
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    l_y_lower = conf['PrimaryGenerator']['YLower']
    l_e_lower = conf['PrimaryGenerator']['ELower']
    l_y_upper = conf['PrimaryGenerator']['YUpper']
    l_e_upper = conf['PrimaryGenerator']['EUpper']


    data_files = args.data_file_name
    train_examples = np.zeros((1, 1, y_bins))
    print(train_examples.shape)
    train_labels = np.zeros((1, l_y_bins, l_e_bins))
    #test_examples = np.zeros((1, 1, y_bins))
    #test_labels = np.zeros((1, l_y_bins, l_e_bins))
    name_string = ""
    for i in range(len(data_files)):
        data_file = data_files[i]
        name_string = name_string+data_file.replace("/", "")
        with np.load(data_file+'.npz') as data:
            print(data['train_data'][:].shape)
            print(data['train_labels'].shape)
            train_examples = np.append(train_examples, data['train_data'][:, int(x_bins/2), np.newaxis], axis = 0 )
            train_labels = np.append(train_labels, data['train_labels'].reshape(-1, l_y_bins, l_e_bins), axis = 0)
            #test_examples = np.append(test_examples, data['test_data'][:, int(x_bins/2), np.newaxis], axis = 0)
            #test_labels = np.append(test_labels, data['test_labels'].reshape(l_y_bins, l_e_bins), axis = 0)
            print(data_file, train_examples.shape, train_labels.shape)
            #plot_preview(train_examples[-1], train_labels[-1, np.newaxis], "last-train-"+data_file.replace("/", "")+".png")
            #plot_preview(test_examples[-1], test_labels[-1, np.newaxis], "last-test-"+data_file.replace("/", "")+".png")


    train_examples = train_examples[1:].astype(float)
    train_labels = train_labels[1:].astype(float)
    #test_examples = test_examples[1:].astype(float)
    #test_labels = test_labels[1:].astype(float)

    print(train_examples.shape, train_labels.shape)
    energy_ranges = np.linspace(l_e_lower, l_e_upper, l_e_bins+1)
    energy_ranges = energy_ranges[0:-1] #lower Riemann sum
    train_labels_summed_y = np.sum(train_labels, axis = 1)
    total_photon_e = np.dot(train_labels_summed_y, energy_ranges)
    train_examples_reshaped = train_examples.reshape(train_examples.shape[0], train_examples.shape[1]*train_examples.shape[2])
    print(train_examples_reshaped.shape)
    total_edep = np.sum(train_examples_reshaped, axis=1)
    ratios = total_photon_e/total_edep #~1.5e7
    print(np.average(total_photon_e))
    print(np.average(total_edep))
    plt.plot(ratios)
    plt.savefig("ratios-"+name_string+".png")


def find_energy_ratio():
    gin = h5py.File('fake-data/response-4500A.h5')
    edeps = gin['compton-electron']['edep'] # (100, 1, 5, 300, 450)
    print(edeps.shape)
    edeps = np.squeeze(edeps)

    num_events = np.squeeze(gin["num_events"]) #100
    print(num_events.shape)
    energy_ranges = gin['i0'] #101
    print(energy_ranges.shape)

    edep_flattened = edeps.reshape(100,-1)
    edep_per_sheet = np.sum(edep_flattened, axis=1)
    print(edep_per_sheet.shape)

    energy_range_lengths = np.diff(energy_ranges)
    energy_midpoints = energy_ranges[0:-1]+energy_range_lengths/2
    print(energy_midpoints.shape)
    energy_per_sheet = num_events*energy_midpoints
    print(energy_per_sheet.shape)

    ratios = edep_per_sheet/energy_per_sheet

    plt.plot(energy_midpoints, ratios)
    plt.savefig("ratios-of-edep-vs-energy.png")


if __name__ == "__main__":
    find_energy_ratio()