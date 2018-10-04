from os.path import basename, dirname, exists, join
import os
import sys
import random

if len(sys.argv) != 3:
    print 'python %s fname num_of_split' % sys.argv[0]
    exit()

fname = sys.argv[1]
num = int(sys.argv[2])

def chunk(xs, n):
    ys = list(xs)
    print 'Random shuffling'
    random.shuffle(ys)
    chunk_length = len(ys) // n
    needs_extra = len(ys) % n
    start = 0
    for i in xrange(n):
        if i < needs_extra:
            end = start + chunk_length + 1
        else:
            end = start + chunk_length
        yield ys[start:end]
        start = end

lines = open(fname, 'r').readlines()
print 'Total num of lines:', len(lines)
if exists(fname + '_ids'):
    idlines = open(fname + '_ids').readlines()
    print 'Total num of id lines:', len(idlines)
    if len(idlines) != len(lines):
        raise RuntimeError('input / ids mismatch')
else:
    idlines = None
if not exists(join(dirname(fname), 'split.%d'%num)):
    os.makedirs(join(dirname(fname), 'split.%d'%num))
count = 0
for i in chunk(range(len(lines)), num):
    name = join(dirname(fname), 'split.%d'%num, basename(fname) + '.' + str(count))
    idname = join(dirname(fname), 'split.%d'%num, basename(fname) + '_ids.' + str(count))
    if exists(name):
        raise RuntimeError("destination file exists")
    print 'Writing %d lines into %s' % (len(i), name)
    if idlines is None:
        with open(name, 'w') as fp:
            for l in i:
                fp.write(lines[l])
    else:
        with open(name, 'w') as fp,\
             open(idname, 'w') as idfp:
            for l in i:
                fp.write(lines[l])
                idfp.write(idlines[l])
    count += 1
