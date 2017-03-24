#!/usr/bin/python
import os
import sys

machfile = ['./singlemach.vm']
nodes=['7']

traindata = ['./20news.train']
numservers = ['3']
numworkerthreads = ['2']

prog=['./bin/medlda ']  
os.system("mpirun -machinefile "+machfile[0]+" --display-map --bynode -np "+nodes[0]+" "+prog[0]
  +" -machfile "+machfile[0]+" -schedulers "+numservers[0]
  +" -num_thread "+numworkerthreads[0]+" -train_prefix "+traindata[0]);
