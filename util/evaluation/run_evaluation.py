#!python
import subprocess
import numpy as np
import sys

# dataset='E:/work/ml-datasets/letor3.0/OHSUMED/QueryLevelNorm'
# dataset = '/mirror/ml/ml-datasets/letor3.0/OHSUMED/QueryLevelNorm';
# dataset = '/mirror/ml/ml-datasets/letor4.0/MQ2007/';
# dataset = '/mirror/ml/ml-datasets/mslr/mslr-web10k';
if len(sys.argv) == 2:
    dataset = sys.argv[1]
    eval_name = 'test'
elif len(sys.argv) == 3:
    dataset = sys.argv[1]
    eval_name = sys.argv[2]
else:
    print 'python run_evaluation.py dataset'
    sys.exit()

if 'mslr' in dataset:
    eval_cmd = 'src/evaluation/evaluate_pl/eval-score-mslr.pl'
else:
    eval_cmd = 'src/evaluation/evaluate_pl/Eval-Score-4.0.pl'

result_map = []
result_ndcg = []
for i in range(5):
    i += 1
    cmd = 'perl %s %s/Fold%d/%s.txt %s/rankelm/%s.fold%d temp 0' % (eval_cmd, dataset, i, eval_name, dataset, eval_name, i);
    cmd = cmd.split(' ')

    subp = subprocess.Popen(cmd)
    subp.wait()
    f = open('temp', 'r')
    result = []
    for l in f.readlines():
        l = l.strip()
        if l == '': continue
        l = l.split()
        if l[0]=='Average':
            result += [[float(m) for m in l[1:]]]

    result_map += [result[0]]
    result_ndcg += [result[1]]

result_ndcg = np.array(result_ndcg)
result_map = np.array(result_map)

print 'qid     NDCG@1  NDCG@2  NDCG@3  NDCG@4  NDCG@5  NDCG@6  NDCG@7  NDCG@8  NDCG@9  NDCG@10 MeanNDCG'
for i,l in enumerate(result_ndcg):
    print 'fold%d  '%(i+1),
    print '  '.join(['%.4f'%m for m in l])

print 'n_mean ',
print '  '.join(['%.4f'%m for m in result_ndcg.mean(0)])
print


print 'qid     P@1     P@2     P@3     P@4     P@5     P@6     P@7     P@8     P@9     P@10    MAP'
for i,l in enumerate(result_map):
    print 'fold%d  '%(i+1),
    print '  '.join(['%.4f'%m for m in l])

print 'm_mean ',
print '  '.join(['%.4f'%m for m in result_map.mean(0)])
