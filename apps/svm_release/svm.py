#!/usr/bin/python
import os
import sys

machfile = ['./singlemach.vm']
inputfile = ['./input/rcv1_train.binary']
dparallel = [' 16 ']
prog = ['./bin/svm-dual ']

os.system(" mpirun -machinefile "+machfile[0]+" "+prog[0]+" --machfile "+machfile[0]+" -input "+inputfile[0]+" -max_iter 200 -C 1.0 -epsilon 0.0001 "+" -parallels "+dparallel[0]+" ");


