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

def reconf(conf, desc, func_index, num_events):
    result = copy.deepcopy(conf)
    result['Detectors']['ComptonScint']['File'] = 'out/' + desc + '.h5'
    result['PrimaryGenerator']['NumEvents'] = num_events
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
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        num_events = 20000000#2e6
        random_affix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        #print(random_affix, "random_affix")
        tasks = []
        conf = toml.load('task1.toml')
        for f_i in range(0, 1):
            desc = 'gYrE-col-2e7-{}-{}'.format(random_affix, f_i)
            tasks.append(compton.Task(
                reconf(conf, desc, f_i, num_events),
                desc, 'pbpl-compton-mc'))
        #data = {'random_affix' : random_affix,
        #        'num_events' : 2000000}
    else:
        #data = None
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
