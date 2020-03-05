# -*- coding: utf-8 -*-
import sys
import subprocess
import time
from tempfile import NamedTemporaryFile
import toml
import asteval
import tqdm
import os
from collections import deque
import numpy as np
from collections import namedtuple
import itertools
import h5py
import Geant4 as g4
from Geant4.hepunit import *

class Task:
    def __init__(self, conf, desc, exec_path, show_bar=True, num_events=None):
        self.conf = conf
        self.desc = desc
        self.exec_path = exec_path
        self.show_bar = show_bar
        self.bar = None
        self.bad_retval = False
        self.conf_filename = None
        self.num_events = num_events
        self.current = 0

    def __del__(self):
        if not self.bad_retval:
            # Task completed successfully. Dispose configuration file.
            if self.conf_filename is not None:
                os.unlink(self.conf_filename)

    def start(self):
        with NamedTemporaryFile('w', delete=False) as f:
            self.conf_filename = f.name
            toml.dump(self.conf, f)
            f.close()
        self.proc = subprocess.Popen(
            [self.exec_path, f.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def update_status(self):
        if self.proc.poll() is not None:
            # process terminated
            self.current = self.num_events
            if self.bar is not None:
                self.bar.update(self.bar.total - self.bar.n)
            if self.proc.poll() != 0:
                # task did not complete successfully.  dump info.
                self.bad_retval = True
                sys.stdout.write('# {}: {} {}\n'.format(
                    self.desc, self.exec_path, self.conf_filename))
                if self.proc.poll() != 0:
                    for x in self.proc.stderr:
                        sys.stdout.write(x.decode('utf-8'))
                sys.stdout.write('\n')
            return False

        if len(self.proc.stderr.peek()) != 0:
            line = self.proc.stderr.readline().decode('utf-8')
            if line[:4] == 'TOT=':
                num_events = int(line[4:])
                self.num_events = num_events
                if self.show_bar:
                    fmt = ('{desc:>30s}:{percentage:3.0f}% ' +
                           '|{bar}| {n_fmt:>9s}/{total_fmt:<9s}')
                    self.bar = tqdm.tqdm(
                        total=num_events, bar_format=fmt, desc=self.desc)
            elif line[:4] == 'CUR=':
                current = int(line[4:])
                self.current = current
                if self.bar is not None:
                    self.bar.update(current - self.bar.n)
        return True

class ParallelTaskRunner:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def run(self):
        for task in self.tasks:
            task.start()
        running_tasks = self.tasks
        while 1:
            for task in running_tasks:
                if task.update_status() == False:
                    running_tasks.remove(task)
            time.sleep(0.2)
            if len(running_tasks) == 0:
                break

class SerialTaskRunner:
    def __init__(self):
        self.tasks = []
        self.max_num_threads = 4

    def add_task(self, task):
        self.tasks.append(task)

    def run(self):
        for task in self.tasks:
            task.start()
        running_tasks = set()
        fmt = ('{desc:>30s}:{percentage:3.0f}% ' +
               '|{bar}| {n_fmt:>9s}/{total_fmt:<9s}')
        bar = tqdm.tqdm(
            total=len(self.tasks), bar_format=fmt)
        while 1:
            for task in list(running_tasks):
                if task.update_status() == False:
                    running_tasks.remove(task)
            time.sleep(0.2)
            while len(self.tasks) and len(running_tasks)<self.max_num_threads:
                new_task = self.tasks.pop()
                running_tasks.add(new_task)
                bar.set_description_str(new_task.desc)
                bar.update(1)
                new_task.start()
            if len(running_tasks) == 0:
                break

def RunMonteCarloSingleIndex(
        conf, reconf, vals, desc, out_filename, total_num_events,
        min_num_events_per_thread, max_num_threads):
    num_threads = max(1, min(
        total_num_events//min_num_events_per_thread, max_num_threads))
    q, r = divmod(total_num_events, num_threads)
    num_events_per_thread = np.ones(num_threads, dtype=int)*q
    num_events_per_thread[0:r] += 1
    assert(total_num_events == num_events_per_thread.sum())

    out_filenames = []
    running_tasks = deque()
    for i, num_events in enumerate(num_events_per_thread):
        with NamedTemporaryFile('w', delete=False) as f:
            out_filenames.append(f.name)

        running_tasks.append(
            Task(
                reconf(conf, *vals, num_events, f.name),
                'none', 'pbpl-compton-mc', False, num_events))
    for task in list(running_tasks):
        task.start()

    fmt = ('{desc:>30s}:{percentage:3.0f}% ' +
           '|{bar}| {n_fmt:>9s}/{total_fmt:<9s}')
    bar = tqdm.tqdm(total=total_num_events, bar_format=fmt, desc=desc)

    finished_sum = 0
    while 1:
        for task in list(running_tasks):
            if task.update_status() == False:
                running_tasks.remove(task)
                finished_sum += task.num_events
        current_running = int(np.array(
            [t.current for t in running_tasks]).sum())
        bar.update(current_running + finished_sum - bar.n)
        time.sleep(0.2)
        if len(running_tasks)==0:
            break

    # merge results
    # Any dataset with 'num_events' attribute is treated as 'data' and
    # is summed in the output.  Otherwise, datasets are treated as 'bins'
    # and are simply copied to the output.

    with h5py.File(out_filename, 'w') as fout:
        for filename in out_filenames:
            num_events = {}
            with h5py.File(filename, 'r') as fin:
                def visit(k, v):
                    if not isinstance(v, h5py.Dataset):
                        return
                    if k not in fout:
                        fin.copy(v, fout, k)
                    else:
                        if 'num_events' in v.attrs:
                            fout[k][()] += v[()]
                            fout[k].attrs['num_events'] += (
                                v.attrs['num_events'])
                        else:
                            assert(np.array_equal(fout[k][()], v[()]))
                fin.visititems(visit)
    for filename in out_filenames:
        os.unlink(filename)


def RunMonteCarlo(
        indices, conf, reconf, out_filename,
        num_events_per_run, min_num_events_per_thread, max_num_threads):
    indices = np.array(indices).T
    indices_shape = [len(x) for x in indices[0]]
    if isinstance(num_events_per_run, int):
        num_events_per_run = num_events_per_run * np.ones(
            indices_shape, dtype=int)

    filenames = {}
    for i in itertools.product(*[range(len(v)) for v in indices[0]]):
        desc = ', '.join(['{}={}'.format(A, B) for A, B in zip(indices[1], i)])
        vals = np.array([indices[0][j][i[j]] for j in range(len(i))])
        f = NamedTemporaryFile('w', delete=False)
        filenames[i] = f.name
        f.close()
        RunMonteCarloSingleIndex(
            conf, reconf, vals,
            desc, f.name, num_events_per_run[i],
            min_num_events_per_thread, max_num_threads)

    aeval = asteval.Interpreter(use_numpy=True)
    for q in g4.hepunit.__dict__:
        aeval.symtable[q] = g4.hepunit.__dict__[q]

    # merge results
    # Any dataset with 'num_events' attribute is treated as 'data' and
    # is summed in the output.  Otherwise, datasets are treated as 'bins'
    # and are simply copied to the output.
    path = os.path.split(out_filename)[0]
    if path != '':
        os.makedirs(path, exist_ok=True)
    with h5py.File(out_filename, 'w') as fout:

        with h5py.File(list(filenames.values())[0], 'r') as fin:
            def visit(k, v):
                if not isinstance(v, h5py.Dataset):
                    return
                if k not in fout and 'num_events' not in v.attrs:
                    fin.copy(v, fout, k)
            fin.visititems(visit)

        for i, (vals, label, unit) in enumerate(indices.T):
            dset_name = 'i{}'.format(i)
            try:
                unit = float(aeval(unit))
                fout[dset_name] = vals/unit
            except TypeError:
                fout[dset_name] = vals
            fout[dset_name].attrs.create('label', np.string_(label))
            fout[dset_name].attrs.create('unit', np.string_(unit))

        num_events = {}
        for i in itertools.product(*[range(len(v)) for v in indices[0]]):
            with h5py.File(filenames[i], 'r') as fin:
                def visit(k, v):
                    if 'num_events' not in v.attrs:
                        return
                    if k not in fout:
                        dset_shape = indices_shape + list(v.shape)
                        dset = fout.create_dataset(
                            k, shape=dset_shape, dtype=float)
                        num_events[k] = np.zeros(indices_shape)
                        dset.attrs.create('unit', np.string_(v.attrs['unit']))
                    fout[k][i] = v
                    num_events[k][i] = v.attrs['num_events']
                fin.visititems(visit)

        the_num_events = list(num_events.values())[0]
        for v in num_events.values():
            assert(np.array_equal(the_num_events, v))
        fout['num_events'] = the_num_events

    for k, v in filenames.items():
        os.unlink(v)
