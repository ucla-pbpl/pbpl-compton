#!/usr/bin/env python
import os
import toml
import sys

os.system('rm -f *wrl *h5')

print('### RUNNING GEANT4 (design.wrl) ###')
# conf = toml.load('pwfa.toml')
# A = conf['PrimaryGenerator']
# A['NumEvents'] = 100

# with open('temp.toml', 'w') as fout:
#     toml.dump(conf, fout)
os.system('pbpl-compton-mc pwfa.toml vis.mac')
#os.system('pbpl-compton-mc temp.toml vis.mac > /dev/null 2>&1')
os.system('pbpl-compton-extrude-vrml g4_00.wrl --radius=0.8 --num-points=8 --output=design.wrl')
os.system('rm -f temp.toml g4*wrl *h5')

# print('### RUNNING GEANT4 (gamma-20MeV.wrl) ###')
# conf = toml.load('pwfa.toml')
# A = conf['PrimaryGenerator']
# A['PythonGenerator'] = 'pwfa.repetitive_spray'
# A['PythonGeneratorArgs'] = ['gamma', '20*MeV', '0*mm', '0*mm', '-10*mm']
# A['NumEvents'] = 10000
# with open('temp.toml', 'w') as fout:
#     toml.dump(conf, fout)
# os.system('pbpl-compton-mc temp.toml vis.mac > /dev/null 2>&1')
# os.system('pbpl-compton-extrude-vrml g4_00.wrl --radius=1.0 --num-points=8 --output=gamma-20MeV.wrl')
# os.system('rm -f temp.toml g4*wrl *h5')
