#!/usr/bin/env python
import io
import os
import sys
import fcntl
import regex as re
import argparse
from os.path import basename, dirname, exists, join
from subprocess32 import Popen, PIPE, STDOUT
from signal import SIGINT, SIGTERM, SIGKILL
from time import clock, sleep
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
np.set_printoptions(linewidth=200, threshold=2000)
from socket import gethostname
hostname = gethostname() + '\n'

STALL_TIMEOUT = 600

def get_args(argv):
    '''
    read command line arguments and process them
    '''
    #mandatory arguments
    helper = argparse.ArgumentParser(description='wrapper to run medlda through mpirun')
    helper.add_argument('-f', '--features', dest='features', type=str, required=True,
                        help='directory for feature files assuming lsvm_train and lsvm_test')
    helper.add_argument('--dumpdir', dest='dumpdir', type=str, required=False, default=None,
                        help='directory for dumping output')
    helper.add_argument('--loaddir', dest='loaddir', type=str, required=False, default=None,
                        help='directory for reading stored model for hot-start')
    helper.add_argument('-a', '--alpha', dest='alpha', type=str, default='.32', required=False,
                        help='alpha prior for topics')
    helper.add_argument('-b', '--beta', dest='beta', type=str, default='.02', required=False,
                        help='beta prior for features (words)')
    helper.add_argument('-c', '--cost', dest='cost', type=str, default='3.2', required=False,
                        help='cost for classifier loss')
    helper.add_argument('-e', '--ell', dest='ell', type=str, default='64', required=False,
                        help='ell parameter on classifier loss')
    helper.add_argument('-p', '--eps', dest='eps', type=str, default='0.1', required=False,
                        help='ell parameter on classifier loss')
    helper.add_argument('-n', '--num_burnin', dest='num_burnin', type=int, required=False,
                        help='number of burnin iterations for training')
    helper.add_argument('-s', '--num_sample', dest='num_sample', type=int, required=False,
                        help='number of sample iterations for training')
    helper.add_argument('-k', '--num_topic', dest='num_topic', type=int, default=24, required=False,
                        help='number of topics')
    helper.add_argument('-t', '--num_target', dest='num_target', type=int, default=0, required=False,
                        help='number of regression targets')
    helper.add_argument('-l', '--num_label', dest='num_label', type=int, required=True,
                        help='number of labels')
    helper.add_argument('-v', '--read_vocab', dest='read_vocab', type=str, default='', required=False,
                        help='path to hashed vocab items to get all workers to read the same vocabulary')
    helper.add_argument('-x', '--num_xtra', dest='num_xtra', type=int, default=0, required=False,
                        help='number of "observable" "topic" "values"')
    helper.add_argument('--task_sizes', dest='label_task_sizes', type=str, default='', required=False,
                        help='label task sizes if more than 1: comma separated list')
    helper.add_argument('--save_interval', dest='save_interval', type=int, default=None, required=False,
                        help='how often to save intermediate results from training')
    helper.add_argument('-m', '--machfile', dest='machfile', type=str, default='multimach.vm',
                        required=False)
    helper.add_argument('--TEST_JUMPS', dest='TEST_JUMPS', default=False, action="store_true", required=False)
    try:
        args = helper.parse_args(argv)
        if args.save_interval is None: args.save_interval = args.num_sample + 1
        if args.dumpdir is None: args.dumpdir = join(args.features, 'dump')
        if args.loaddir is None: args.loaddir = join(basename(args.dumpdir), 'load')
    except Exception, e:
        print "Exception: " + str(e)
        helper.print_help()
        sys.exit(1)
    print("The arguments are", args)
    return args


args = get_args(sys.argv[1:])
nodes = len(open(args.machfile).read().strip().splitlines())
numservers       = nodes // 2
numworkerthreads = 16
numworkers = nodes - numservers - 1

exitline  = re.compile(r'^Rank \((\d+)\) Ready for exit program from main function in ldall.cpp')
abortline = re.compile(r'^\*\*\* Aborted')

def mpirun(cmd, testfile, retries=4, retry_delay=0):
    if exists(testfile): return True
    if not exists(testfile + '_work'):
        open(testfile + '_work', 'wb').write(hostname)
        sleep(1)
    if open(testfile + '_work', 'rb').read() != hostname:
        return False
    while retries > 0:
        retries -= 1
        workers_finished = set()
        try:
            print nm, cmd
            if 'LD_LIBRARY_PATH' in os.environ:
                env = dict(os.environ)
                del env['LD_LIBRARY_PATH']
            else:
                env = os.environ
            proc = Popen(cmd, bufsize=1, stdin=None, stdout=PIPE, stderr=STDOUT, env=env)
            fd = proc.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK
            fcntl.fcntl(fd, fcntl.F_SETFL, fl)
            pout = io.open(fd)
            fc = lc = clock()
            while proc.returncode is None:
                for line in pout:
                    lc = clock()
                    sys.stderr.write(line)
                    m = abortline.match(line)
                    if m is not None:
                        sys.stderr.write('FOUND ' + m.group(0) + '\n')
                        sleep(5)
                        proc.terminate()
                        break
                    m = exitline.match(line)
                    if m is not None:
                        workers_finished.add(int(m.group(1)))
                        sys.stderr.write('FOUND %d %s: '% (len(workers_finished), sorted(workers_finished)) + m.group(0) + '\n')
                        if len(workers_finished) == numworkers:
                            proc.send_signal(SIGINT)
                            sys.stderr.write('signal sent\n')
                            sleep(1)
                proc.poll()
                if lc - fc < 2 * STALL_TIMEOUT:
                    if clock() - lc > STALL_TIMEOUT:
                        sys.stderr.write('*** process stalled for %s seconds ***\n' % STALL_TIMEOUT)
                        proc.send_signal(SIGINT)
                        sys.stderr.write('signal sent\n')
                        sleep(1)
                else:
                    sleep(1e-5)
        except:
            if proc.returncode is None:
                proc.terminate()
            pass
        if len(workers_finished) == numworkers:
            sys.stderr.write('return True!\n')
            os.unlink(testfile+'_work')
            return True
        sys.stderr.write('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        sys.stderr.write('!!!!!!!!!!!! R E T R Y !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        sys.stderr.write('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        sleep(retry_delay)
    os.unlink(testfile+'_work')
    return False

nm               = 'medldaR_%s_%s_a%s_b%s_c%s_l%s_ml%d_t%se%s_%s_%s' % (args.label_task_sizes, args.num_topic, args.alpha, args.beta, args.cost,
                                                                        args.ell, args.num_label, args.num_target, args.eps, args.num_burnin, args.num_sample)

traindata        = '%s/split.%d/lsvm_train' % (args.features, numworkers)
testdata         = '%s/split.%d/lsvm_test' % (args.features, numworkers)
dumpdata         = '%s/%s/' % (args.dumpdir, nm)
loaddata         = '%s/%s/' % (args.loaddir, nm)

prog = './bin/medlda'
cmd = ["mpirun", "-machinefile", args.machfile, "--display-map", "--map-by", "node", "-np", str(nodes), prog,
       "-machfile", args.machfile, "-schedulers", str(numservers), "-num_thread", str(numworkerthreads), "-read_vocab", args.read_vocab,
       "-num_burnin", str(args.num_burnin), "-num_sample", str(args.num_sample), "-save_interval", str(args.save_interval),
       "-num_topic", str(args.num_topic), "-num_label", str(args.num_label), "-num_xtra", str(args.num_xtra), "-label_task_sizes", args.label_task_sizes,
       "-num_target", str(args.num_target), "-alpha", args.alpha, "-beta", args.beta, "-cost", args.cost, "-ell", args.ell, "-dump_prefix", dumpdata, "-eps", args.eps]
if args.TEST_JUMPS: cmd.extend(["-TEST_DSFMT_JUMP_STATE", "1"])
train_cmd = cmd + ["-train_prefix", traindata]
test_cmd = cmd + ["-test_prefix", testdata]

if not exists(dumpdata + '_model'):
    if not exists(traindata + '.0'):
        os.system('python ./split.py %s %d' % (join(dirname(dirname(traindata)), basename(traindata)), numworkers))
    if not exists(traindata + '.0'):
        sys.stderr.write('could not create training data splits at %s\n' % traindata)
        sys.exit(-1)

    if not exists(dirname(dumpdata)): os.makedirs(dirname(dumpdata))

    res = mpirun(train_cmd, dumpdata + '_model')
    sys.stderr.write('res is %s for train %s\n' % (res, nm))
    if not res:
        sys.exit(-1)

if not exists(dumpdata + '_test_pred.0'):
    if not exists(testdata + '.0'):
        os.system('python ./split.py %s %d' % (join(dirname(dirname(testdata)), basename(testdata)), numworkers))
    if not exists(testdata + '.0'):
        sys.stderr.write('could not create testing data splits at %s\n' % traindata)
        sys.exit(-1)
    res = mpirun(test_cmd, dumpdata + '_test_pred.0')
    sys.stderr.write('res is %s for test %s\n' % (res, nm))
    if not res:
        sys.exit(-1)
