from sys import stdin
from sys import argv
import shlex, subprocess

########## SETTINGS ########## 
# MPI and EXEC
MPI = 'mpirun'
EXECPATH = argv[1]
EXEC = EXECPATH + '/pgbrt'

# DATA
DATAPATH = argv[2]

if argv[3] == 'm':
    # MSFT FOLD 1 DATA
    FEATURES = 137
    TRAIN = DATAPATH + '/train.txt'
    TRAINSIZE = 723412
    VALID = DATAPATH + '/vali.txt'
    VALIDSIZE = 235259
    TEST = DATAPATH + '/test.txt'
    TESTSIZE = 241521
elif argv[3] == "y":
    # YAHOO SET 1 DATA
    FEATURES = 586
    TRAIN = DATAPATH + '/set1.train.con.txt'
    TRAINSIZE = 473134
    VALID = DATAPATH + '/set1.valid.con.txt'
    VALIDSIZE = 71083
    TEST = DATAPATH + '/set1.test.con.txt'
    TESTSIZE = 165660
else:
    exit()

# PGBRT SETTINGS
PROCS = int(argv[4])
DEPTH = 4
TREES = int(argv[5])
RATE = 0.06


########## RUN PGBRT ########## 
# run training command
cmdline = \
    '%s -np %d %s ' %  (MPI, PROCS, EXEC) + \
    '%s %d %d %d %d %f ' % (TRAIN, TRAINSIZE, FEATURES, DEPTH, TREES, RATE) + \
    '-V %s -v %d -m -t' % (VALID, VALIDSIZE)
args = shlex.split(cmdline);
p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# open log file
logfile = 'pgbrt.%s.%d.%d.log' % (argv[3], PROCS, TREES)
log = open(logfile, 'w')
log.write('#%s\n' % cmdline)
log.flush()

# print results to log file
for l in p.stdout:
    log.write(l)
    log.flush()

# close log file
log.close()
