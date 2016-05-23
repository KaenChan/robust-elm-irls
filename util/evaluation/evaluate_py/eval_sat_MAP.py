from sys import argv
from evtools import evaluate, rel2ranks, rel2labels
import numpy as np

def printhelp():
    print(
'''usage:  python eval_sat_MAP.py TEST_FILE PRED_FILE
    ''')

# readtargets
def readtargets(filename):
    labels = []
    queries = []
    for line in open(filename):
        y,q,rest = line.split(' ',2)
        labels += [float(y)]
        qid = q.split(':')[1]
        queries += [int(qid)]
    return (labels,queries)

# readpreds
def readpreds(filename):
    preds = []
    for line in open(filename):
        preds += [float(line.strip())]
    return preds


# if -h in args, print help
if '-h' in argv:
    printhelp()
    exit()

# check args
if len(argv) != 3:
    print('illegal usage, incorrect number of arguments')
    printhelp()
    exit()

# get args    
testfile = argv[1]
predfile = argv[2]

# read targets and predictions
labels,queries = readtargets(testfile)
preds = readpreds(predfile)

ranks=rel2ranks(preds,queries);
ranks = np.array([[int(m) for m in l.split(' ')] for l in ranks])
# print ranks
print ranks.shape

labs=rel2labels(labels,queries);
labs = np.array([[int(float(m)) for m in l.split(' ')] for l in labs])
# print labs
print labs.shape

#print len(ranks),len(labs);
assert(len(ranks)==len(labs));

b = np.array(ranks).argmin(axis=1)
r = np.array([labs[i][j] for i,j in enumerate(b)])
# print r
nsolved = sum(r!=0)
MAP1 = (nsolved*1.0/r.shape[0])
print 'MAP1', MAP1


for idx in range(1,31):
    b = np.array(ranks).argsort(axis=1)
    r2 = np.array([labs[i][j] for i,j in enumerate(b[:,idx])])

    r1 = (r!=0)
    r2 = (r2!=0)
    r = r1 + r2
    nsolved = sum(r!=0)
    MAP2 = (nsolved*1.0/r.shape[0])
    print 'MAP%d'%(idx+1), MAP2

