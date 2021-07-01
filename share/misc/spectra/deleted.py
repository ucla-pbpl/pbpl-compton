import numpy as np
#The Monte Carlo run consisted of 1.0, 1.8, 2.0, and 3.0 MeV gammas.  
#There were 1e9 of each, for a total of 4e9 gammas simulated.

def main():
    photon_energies = [1, 1.8, 2, 3]
    photon_number = [1e9, 1e9, 1e9, 1e9]
    width = 0.001
    
    photon_e_bins = np.logspace(np.log(0.01), np.log(7), 129, base=np.e)
    print(photon_e_bins)
    log_e = np.interp(photon_e_bins, xp, fp)
    #log_e = np.append(0, log_e)
    #photon_e_bins = np.append(0.16, photon_e_bins)
    ax.plot(photon_e_bins, log_e)
            
    np.savez("darpa.npz", photon_bins = photon_e_bins, photon_energy = log_e)

if __name__ == '__main__':
    sys.exit(main())
