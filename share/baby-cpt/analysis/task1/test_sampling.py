import numpy as np
import matplotlib.pyplot as plt



l_e_lower = 10
l_e_upper = 20
l_e_bins = 20
num_simulations = 10
energy_ranges = np.array([10, 11, 12, 14, 14.5, 16, 16.2, 16.9, 17, 19.5, 20])
#energy_ranges = np.logspace(np.log(l_e_lower), np.log(l_e_upper), num_simulations+1, base=np.e)
print("what is given")
print( energy_ranges)
resampled_energy_ranges = np.logspace(np.log(l_e_lower), np.log(l_e_upper), l_e_bins+1, base=np.e)
print("what i want")
print(resampled_energy_ranges)
e_weights = np.ones(l_e_bins)
#want this to be energy_density. 
#amount of energy between i MeV, j MeV is energy_density((i+j)/2)*(j-i)
#print(e_weights)
weights = np.zeros(num_simulations)#, dtype = int
for j in range(0, l_e_bins):
    ej_l = np.exp(np.log(l_e_lower)+(np.log(l_e_upper)-np.log(l_e_lower))/l_e_bins*(j))
    ej_u = np.exp(np.log(l_e_lower)+(np.log(l_e_upper)-np.log(l_e_lower))/l_e_bins*(j+1))
    ej = np.random.uniform(low=ej_l, high=ej_u)
    #print(ej)
    idx_l = np.where(energy_ranges<ej)
    idx_u = np.where(energy_ranges>ej)
    print("ej=", ej)
    print("idx_l", idx_l)
    print("idx_u", idx_u)
    if(len(idx_l[0])==0 or len(idx_u[0])==0):
        continue
    else:
        #print("idx[-1][-1]", idx_l[-1][-1], idx_u[0])
        conversion = (ej_u-ej_l)/(energy_ranges[idx_u[-1][0]]-energy_ranges[idx_l[-1][-1]]) 
        print("conversion", conversion)
        weights[idx_l[-1][-1]]+=e_weights[j]*conversion
another_weights = np.interp(energy_ranges[:-1], resampled_energy_ranges[0:-1], e_weights)
plt.scatter(resampled_energy_ranges[0:-1], e_weights, label="what I want")
plt.scatter(energy_ranges[:-1], weights, label="corresponds to")
plt.scatter(energy_ranges[:-1], another_weights, label="corresponds to interp")
plt.ylabel("energy_density")
plt.xlabel("energy")
plt.legend()
plt.show()