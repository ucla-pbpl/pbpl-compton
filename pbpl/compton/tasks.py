# -*- coding: utf-8 -*-
import sys
import subprocess
from mpi4py import MPI
import time
from tempfile import NamedTemporaryFile
import toml
import tqdm
import os

class Task:
    def __init__(self, conf, desc, exec_path):
        self.conf = conf
        self.desc = desc
        self.exec_path = exec_path
        self.bar = None
        self.bad_retval = False
        self.conf_filename = None
        self.proc = None

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
                fmt = ('{desc:>30s}:{percentage:3.0f}% ' +
                       '|{bar}| {n_fmt:>9s}/{total_fmt:<9s}')
                self.bar = tqdm.tqdm(
                    total=num_events, bar_format=fmt, desc=self.desc)
            elif line[:4] == 'CUR=':
                current = int(line[4:])
                self.bar.update(current - self.bar.n)
        return True

class MPITaskRunner:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def worker(self, task_index):
        if task_index is None or task_index<0 or task_index>=len(self.tasks):
            return "invalid/empty task"
        #subprocess.call("pbpl-compton-mc test.toml", shell=True)
        print(task_index, " gets to start")
        #p = subprocess.Popen(
        #    ["python", "test_multi_proc.py"],
        #    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.tasks[task_index].start()
        p = self.tasks[task_index].proc
        if p is None:
            print(task_index, " Popen returns None")
        #
        print(task_index, p, " proc in worker")
        print(task_index, self.tasks[task_index].desc, "desc in worker")
        return self.tasks[task_index].desc

    def run(self):  
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        start_index = 0
        #while(start_index<=len(self.tasks)):
        if rank == 0:
            # Here should normally be a list with many more entries, subdivided
            # among all the available cores. I'll keep it simple here, so one has
            # to run this script with mpirun -np 2 ./script.py
            #if(len(self.tasks)>=size):
                task_indices_to_run = [start_index+i for i in range(size)]
            #else:
                #tasks_to_run = self.tasks+([None]*(size-len(self.tasks)))
                #self.tasks = []
        else:
            task_indices_to_run = None

        task_index = comm.scatter(task_indices_to_run, root=0)
        res = self.worker(task_index)
        results = comm.gather(res, root=0)

        #print (res) #None
        print (results)
        #running_task_indices = task_indices_to_run

        while 1:
            #print(task_index, " task_index in status check")
            print(task_index, self.tasks[task_index].proc, " proc in status check")
            if self.tasks[task_index].proc is None:
                #running_task_indices.remove(task_index)
                print("Task not started: ", task_index)
                break
            if self.tasks[task_index].update_status() == False:
                break
            time.sleep(0.2)
        #print(running_task_indices)
        #while running_task_indices is not None:
        #    for task_index in running_task_indices:
        #        print(task_index, " task_index in status check")
        #        print(self.tasks[task_index].proc, " proc in status check")
        #        if self.tasks[task_index].proc is None:
        #            running_task_indices.remove(task_index)
        #            print("Task not started: ", task_index)
        #            continue
        #        if self.tasks[task_index].update_status() == False:
        #            running_task_indices.remove(task_index)
        #    time.sleep(0.2)
        #    if len(running_task_indices) == 0:
        #        break
        

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
