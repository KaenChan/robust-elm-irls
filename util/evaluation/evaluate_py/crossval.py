'''
Copyright (c) 2011, Washington University in St. Louis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY 
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from sys import argv
from sys import stdin

# help menu
def printhelp():
    print(
'''usage:  python crossval.py VAL_METRIC_INDEX [TEST_METRIC_INDEX] [-r]
  perform cross-validation on metrics/trees read from stdin, using 
  metric[VAL_METRIC_INDEX] and assuming smaller values are better 
  unless option -r is set; prints corresponding test metric if 
  TEST_METRIC_INDEX is included, otherwise prints trees
    ''')

# remove executable name from args
argv.pop(0)

# if no args, print help
if len(argv) < 1:
    print('illegal usage, too few argmuments')
    printhelp()
    exit()
    
# if too many args, print help
if len(argv) > 3:
    print('illegal usage, too many argmuments')
    printhelp()
    exit()

# if -h in args, print help
if '-h' in argv:
    printhelp()
    exit()

# determine if larger or smaller is better
bigsmall = 1
if '-r' in argv:
    bigsmall = -1
    argv.remove('-r')

# check for cross-validation metric
if len(argv) < 1:
    print('illegal usage, no validation metric index')
    printhelp()
    exit()

# get index of desired cross-validation metric
vi = int(argv[0])
if vi < 0:
    print('illegal usage, bad validation metric index %s' % argv[0])
    printhelp()
    exit()

# get index of desired test-metric, or decide to print trees
printtrees = (len(argv) != 2);
if not printtrees:
    ti = int(argv[1])
    if ti < 0:
        print('illegal usage, bad test metric index %s' % argv[1])
        printhelp()
        exit()

# store trees until a better metric value is found
trees = []

# store best metric value
bestiteration = 0
bestmetric = 0
besttestmetric = 0
firstmetric = False
i = -1

for line in stdin:
    # skip comment lines
    if line.startswith('#'): continue
    
    # if tree, add to list
    if ':' in line:
        if printtrees: trees += [line]
        continue
    
    # get metric values
    i += 1;
    metrics = line.strip().split(',')
    if (vi >= len(metrics)):
        print('validation metric index too large')
        print line
        print metrics
        printhelp()
        exit()
    
    # check for better metric
    metric_i = float(metrics[vi])
    if bigsmall * metric_i < bigsmall * bestmetric or not firstmetric:
        # store new best metric
        firstmetric = True
        bestmetric = metric_i
        bestiteration = i
        
        # print trees
        if printtrees:
            for t in trees: print(t.strip())
            trees = []
        
        # or store test metric
        else:
            besttestmetric = float(metrics[ti])

# print metrics, if not printing trees
if not printtrees:
    print('%d,%f,%f' % (bestiteration, bestmetric, besttestmetric))
