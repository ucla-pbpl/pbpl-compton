#!/usr/bin/env python
import os
import toml
import sys
import tqdm

fmt = '{desc:>30s} {percentage:3.0f}% |{bar}| {n_fmt:>9s}/{total_fmt:<9s}'
bar = tqdm.tqdm([4500, 9000, 18000], bar_format=fmt)
for current in bar:
    bar.set_description('current = {} A'.format(current))
    with open('calc-energy-scale.toml', 'r') as fin:
        conf_str = fin.read()
    conf_str = conf_str.replace('4500', str(current))
    conf = toml.loads(conf_str)
    with open('temp.toml', 'w') as fout:
        toml.dump(conf, fout)
    os.system('pbpl-compton-calc-energy-scale temp.toml')
os.system('rm -f temp.toml')
