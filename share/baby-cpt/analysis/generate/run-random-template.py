#!/usr/bin/env python
import sys
import copy
import toml
import numpy as np
from pbpl import compton
from Geant4.hepunit import *
import random
import string
from mpi4py import MPI
import time
import subprocess
import argparse

def reconf(conf, desc, func_index):
    result = copy.deepcopy(conf)
    result['Detectors']['ComptonScint']['File'] = 'out/' + desc + '.h5'
    num_events = result['PrimaryGenerator']['NumEvents']
    result['PrimaryGenerator']['PythonGeneratorArgs'] = [
        '{}'.format(num_events), '\'{}\''.format(desc), 
        result['PrimaryGenerator']['YBins'],
        result['PrimaryGenerator']['YLower'],
        result['PrimaryGenerator']['YUpper'],
        result['PrimaryGenerator']['EBins'],
        result['PrimaryGenerator']['ELower'],
        result['PrimaryGenerator']['EUpper']]
    with open('test.toml', 'w') as f:
        toml.dump(result, f)
    return result

def main():
    parser = argparse.ArgumentParser(
        description='Run geant4 simulation based on config')
    parser.add_argument("--config", required=True, nargs=1, 
        help="path to the configuration toml file")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        args = parser.parse_args()
        conf = toml.load(args.config)
        num_events = conf['PrimaryGenerator']['NumEvents']#2e7
        num_simulations = conf['Simulation']['NumSimulations']
        random_affix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        #print(random_affix, "random_affix")
        tasks = []
        
        for f_i in range(0, num_simulations):
            desc = 'taggy-col-{:1e}-{}-{}'.format(num_events, random_affix, f_i)
            tasks.append(compton.Task(
                reconf(conf, desc, f_i),
                desc, 'pbpl-compton-mc'))
    else:
        tasks=None
    task = comm.scatter(tasks, root=0)
    print(task.desc, " gets to start")
    #task.start()
    args=type('', (), {})()
    args.conf = task.conf
    args.macro_filenames = []
    compton.run_main(args)
    #task.proc=p
    #if p is None:
    #    print(task.desc, " Popen returns None")
        #
    #print(task.desc, p, " proc in worker")
    
    #tr = compton.ParallelTaskRunner()
    #tr = compton.MPITaskRunner()

if __name__ == '__main__':
    sys.exit(main())
