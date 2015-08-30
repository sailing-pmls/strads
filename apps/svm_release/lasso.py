#!/usr/bin/python
import os
import sys

machfile = ['./singlemach.vm']

# data setting 
inputfile = ['./input/breast-cancer.txt']

csample = [' 500 ']
column = [' 1000 ']

# scheduler setting
cscheduler = [' 1 ']
scheduler_threads = [' 1 ']

# worker thread per machine
worker_threads = [' 1 ']

# degree of parallelism
set_size = [' 1 ']

prog = ['./bin/svm-dual ']

os.system(" mpirun -machinefile "+machfile[0]+" "+prog[0]+" --machfile "+machfile[0]+" -threads "+worker_threads[0]+" -input "+inputfile[0]+" -schedule_size "+set_size[0]+" -max_iter 100 -C 0.1 -scheduler "+cscheduler[0]+" -threads_per_scheduler "+scheduler_threads[0]+" ");


