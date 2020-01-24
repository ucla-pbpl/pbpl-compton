#!/usr/bin/env python
import os
import toml
import sys

os.system('rm -f *wrl *h5')

# print('### RUNNING GEANT4 (design.wrl) ###')
# conf = toml.load('sfqed.toml')
# A = conf['PrimaryGenerator']
# A['PythoGenerator'] = 'sfqed.pattern_spray'
# A['NumEvents'] = 100
# with open('temp.toml', 'w') as fout:
#     toml.dump(conf, fout)
# os.system('pbpl-compton-mc temp.toml vis.mac > /dev/null 2>&1')
# os.system('pbpl-compton-extrude-vrml g4_00.wrl --radius=0.2 --num-points=8 --output=design.wrl')
# os.system('rm -f temp.toml g4*wrl')

print('### RUNNING GEANT4 (gamma-10MeV.wrl) ###')
conf = toml.load('sfqed.toml')
A = conf['PrimaryGenerator']
A['PythonGenerator'] = 'pbpl.compton.generators.repeater'
A['PythonGeneratorArgs'] = ['gamma', '10*MeV', '[0,0,-100*mm]', '[0,0,1]']
A['NumEvents'] = 20000
with open('temp.toml', 'w') as fout:
    toml.dump(conf, fout)
os.system('pbpl-compton-mc temp.toml vis.mac')
#os.system('pbpl-compton-mc temp.toml vis.mac > /dev/null 2>&1')
os.system('pbpl-compton-extrude-vrml g4_00.wrl --radius=0.8 --num-points=8 --output=gamma-10MeV.wrl')
# os.system('rm -f temp.toml g4*wrl')


# print('### RUNNING GEANT4 (gamma-2GeV.wrl) ###')
# conf = toml.load('sfqed.toml')
# A = conf['PrimaryGenerator']
# A['PythonGenerator'] = 'pbpl.compton.generators.repeater'
# A['PythonGeneratorArgs'] = ['gamma', '2*GeV', '[0,0,-100*mm]', '[0,0,1]']
# A['NumEvents'] = 20000
# with open('temp.toml', 'w') as fout:
#     toml.dump(conf, fout)
# os.system('pbpl-compton-mc temp.toml vis.mac')
# #os.system('pbpl-compton-mc temp.toml vis.mac > /dev/null 2>&1')
# os.system('pbpl-compton-extrude-vrml g4_00.wrl --radius=0.8 --num-points=8 --output=gamma-2GeV.wrl')
# os.system('rm -f temp.toml g4*wrl')
