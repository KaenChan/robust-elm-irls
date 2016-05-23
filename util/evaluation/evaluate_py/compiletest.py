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

from sys import stdin
from sys import argv
import shlex, subprocess

# print help
def printhelp():
    print(
'''usage:  python compiletest.py EXEC_FILE [COMPILER]
  reads trees in a GBRT ensemble from stdin, writes a c_rho++
  program to EXEC_FILE.cpp, and compiles an executable to
  EXEC_FILE.  Optionally a COMPILER may be supplied; default
  compiler string is "g++ -O3".
    ''')

# set compiler
compiler = 'g++ -O3'
if len(argv) == 3:
    compiler = argv.pop(0)
    
# check args
if len(argv) != 2:
    print('Illegal usage. (If supplying a compiler command, wrap in quotes.)')
    printhelp();
    exit();

# open cpp file
binfile = argv[1]
cppfile = binfile + '.cpp'
f = open(cppfile, 'w')

# print header
f.write("""
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <stdio.h>

using namespace std;

double test(float* instance);

void clearinstance(float* instance, int numfeatures) {
    for (int f=0; f<numfeatures; f++)
        instance[f] = 0.f;
}

bool parseFeatureValue(string &cfeature, string &cvalue) {
    // get token
    char* tok;
    if (not (tok = strtok(NULL, " \\n"))) // line stored in state from previous call to strtok
        return false;
    
    // tok is feature:value
	string bit = tok;
	int colon_index = bit.find(":");
	cfeature = bit.substr(0, colon_index);
	cvalue = bit.substr(colon_index+1,bit.length()-colon_index-1);
	
    return true;
}

bool readinstance(int numfeatures, float* instance) {
    // get line from stdin
    string strline;
	getline(cin, strline);
	char* line = strdup(strline.c_str());

	// extract and ignore label (first item)
	strtok(line, " ");

	// get qid, if present, and ignore
	string cfeature, cvalue;
    string qidstr ("qid");
    if (not parseFeatureValue(cfeature, cvalue)) return true;
    if (qidstr.compare(cfeature)) // qid is present
        if (not parseFeatureValue(cfeature, cvalue)) return true;
	
	// get feature values
    int feature = -1;
    float value = -1.f;
	do {
	    // parse and check feature index
	    feature = atoi(cfeature.c_str());
        if (feature < 0) return false;  // obviously invalid
        if (feature >= numfeatures) return true;
            // could be invalid, but most likely the trees found no use for features >= numfeatures
            // and we can skip the remaining feature values for this instance since they are expected
            // to be listed in ascending order by feature index
        
        // store feature value
		value = (float) atof(cvalue.c_str());
        instance[feature] = value;
    } while (parseFeatureValue(cfeature, cvalue));

	// clean up
	free(line);
	
	// return
    return true;
}

void driver(int numfeatures) {
    // variables
    float* instance = new float[numfeatures];
    clearinstance(instance, numfeatures);
    
	// evaluate all data instances
	while (readinstance(numfeatures, instance) and not cin.eof()) {
        // test instance
        double result = test(instance);
        
        // print result
        printf("%f\\n", result);
        clearinstance(instance, numfeatures);
    }
}
""")

# print tree functions
numfeatures = 0
numtrees = 0

def printNode(nodes):
     # parse node
     node = nodes.pop(0)
     feature,split,label = node.split(':')
     feature = int(feature)
     split = float(split)
     label = float(label)
     
     # update numfeatures
     global numfeatures;
     numfeatures = max(numfeatures,feature)
     
     # print splitting node
     if feature >= 0:
         n1 = printNode(nodes)
         n2 = printNode(nodes)
         return 'if (instance[%d] < %f) {%s} else {%s}' % (feature, split, n1, n2)
     
     # or print label node
     else:
         return 'return %f;' % label

for line in stdin:
    numtrees += 1
    nodes = line.strip().split(',')
    f.write('double tree_%d(float* instance) {%s}\n' % (numtrees-1, printNode(nodes)))

# print test function
f.write('double test(float* instance) {\n')
f.write('\tdouble pred = 0.f;\n')
for i in range(numtrees):
    f.write('\tpred += tree_%d(instance);\n' % i);
f.write('\treturn pred;\n')
f.write('}\n')

# print main function
f.write('int main(int argc, char* argv[]) { driver(%d); }\n' % (numfeatures+1))

# close cpp file
f.close()

# compile cpp file
cmdline = '%s %s -o %s' %  (compiler,cppfile,binfile)
args = shlex.split(cmdline);
p = subprocess.call(args)
