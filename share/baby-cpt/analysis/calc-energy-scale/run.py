#!/usr/bin/env python
import os
import toml
import sys

for current in [4500, 9000, 18000]:
    with open('calc-energy-scale.toml', 'r') as fin:
        conf_str = fin.read()
    conf_str = conf_str.replace('4500', str(current))
    conf = toml.loads(conf_str)
    with open('temp.toml', 'w') as fout:
        toml.dump(conf, fout)
    os.system('pbpl-compton-calc-energy-scale temp.toml')
os.system('rm -f temp.toml')
