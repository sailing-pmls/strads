#!/usr/bin/python
import os
import sys

machfile = ['./singlemach.vm']
nodes=['6']

# data setting 
datafilex = ['./input/lasso500by1K.X.mmt']
datafiley = ['./input/lasso500by1K.Y.mmt']
csample = [' 500 ']
column = [' 1000 ']

# scheduler setting
cscheduler = [' 2 ']
scheduler_threads = [' 1 ']

# worker thread per machine
worker_threads = [' 1 ']

# degree of parallelism
set_size = [' 1 ']

prog = ['./bin/cdsolver ']

os.system(" mpirun -machinefile "+machfile[0]+" --display-map --bynode -np "+nodes[0]+" "+prog[0]+" --machfile "+machfile[0]+" -threads "+worker_threads[0]+" -samples "+csample[0]+" -columns "+column[0]+" -data_xfile "+datafilex[0]+" -data_yfile "+datafiley[0]+"  -schedule_size "+set_size[0]+" -max_iter 30000 -lambda 0.001 -scheduler "+cscheduler[0]+" -threads_per_scheduler "+scheduler_threads[0]+" --weight_sampling=false -check_interference=false -algorithm lasso --ring4scheduler=true");
