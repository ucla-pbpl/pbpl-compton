import numpy as np 
import toml
import argparse

#np.savez(output+".npz", photon_bins = photon_e_bins, photon_energy = log_e)

parser = argparse.ArgumentParser(
    description='convert d2W to label (step 2). Step 1 is in get_pwfa_spectra.py')
parser.add_argument("--config", required=True,
    help="the config toml file")
parser.add_argument("--name", required=True,
    help="name the data file")

args = parser.parse_args()

with np.load(args.name+'.npz') as data:
    bins = data['photon_bins']
    energies = data["photon_energy"]
    conf = toml.load(args.config)
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    l_y_lower = conf['PrimaryGenerator']['YLower']
    l_e_lower = conf['PrimaryGenerator']['ELower']
    l_y_upper = conf['PrimaryGenerator']['YUpper']
    l_e_upper = conf['PrimaryGenerator']['EUpper']
    histo = np.zeros([l_y_bins, l_e_bins])
    label_e_bins = np.logspace(np.log(l_e_lower), np.log(l_e_upper), l_e_bins+1, base=np.e)

    histo[int(l_y_bins/2)]=np.interp(label_e_bins[:-1], bins, energies)
    np.savez("output.npz", histo= histo)
