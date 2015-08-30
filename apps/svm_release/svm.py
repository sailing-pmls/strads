#!/usr/bin/python
import os
import sys

machfile = ['./singlemach.vm']

# data setting 
inputfile = ['./input/rcv1_train.binary']

# scheduler setting
cscheduler = [' 1 ']
scheduler_threads = [' 1 ']

# worker thread per machine
worker_threads = [' 1 ']

# degree of parallelism
dparallel = [' 16 ']

prog = ['./bin/svm-dual ']

os.system(" mpirun -machinefile "+machfile[0]+" "+prog[0]+" --machfile "+machfile[0]+" -threads "+worker_threads[0]+" -input "+inputfile[0]+" -max_iter 200 -C 1.0 -scheduler "+cscheduler[0]+" -threads_per_scheduler "+scheduler_threads[0]+" -epsilon 0.0001 "+" -parallels "+dparallel[0]+" ");


