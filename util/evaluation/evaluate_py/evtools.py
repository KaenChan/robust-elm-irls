# From Ananth Mohan's rt-rank distribution
# https://sites.google.com/site/rtranking/home

import os,sys,operator;
from math import log,sqrt
from numpy import log2
from itertools import izip

def evaluate_submission(labels,ranks,k=10,max_grade=2):
# From Yahoo's Learning to Rank Challenge
# http://learningtorankchallenge.yahoo.com/evaluate.py.txt
 """
 Script to compute the NDCG and ERR scores of a submission.
 Labels is a list of lines containing the relevance labels (one line per query).
 Ranks is a list of lines with the predicted ranks (again one line per query). 
 The first integer in a line is the rank --in the predicted ranking--  of the first document, where first refers to the order of the data file.
 k is the truncation level for NDCG
 It returns the mean ERR and mean NDCG
 """
 nq = len(labels) # Number of queries
 assert len(ranks)==nq, 'Expected %d lines, but got %d.'%(nq,len(ranks))
 err = 0.0
 ndcg = 0.0
 for i in range(nq):

  l = [int(float(x)) for x in labels[i].split()]
  try:
    r = [int(x) for x in ranks[i].split()]
  except ValueError:
    raise ValueError('Non integer value on line %d'%(i+1))

  nd = len(l) # Number of documents
  assert len(r)==nd, 'Expected %d ranks at line %d, but got %d.'%(nd,i+1,len(r))

  gains = [-1]*nd # The first element is the gain of the first document in the predicted ranking
  assert max(r)<=nd, 'Ranks on line %d larger than number of documents (%d).'%(i+1,nd)
  for j in range(nd):
    gains[r[j]-1] = (2**l[j]-1.0)
  assert min(gains)>=0, 'Not all ranks present at line %d.'%(i+1)

  p = 1.0
  for j in range(nd):
      r = gains[j]/(2**max_grade)
      err += p*r/(j+1.0)
      p *= 1-r

  dcg = sum([g/log(j+2) for (j,g) in enumerate(gains[:k])])
  gains.sort()
  gains = gains[::-1]
  ideal_dcg = sum([g/log(j+2) for (j,g) in enumerate(gains[:k])])

  if ideal_dcg:
      ndcg += dcg / ideal_dcg
  else:
      ndcg += 1.0

 return (err/nq, ndcg/nq)


def rel2ranks(rels,qs):
    output=[]
    curid=''
    table={}

    def printtable(table):
        inds=table.items();
        inds.sort(key=operator.itemgetter(1),reverse=True);
        pos=dict(zip(map(operator.itemgetter(0),inds),range(1,len(inds)+1)));
        return(' '.join([str(pos[i]) for i in range(1,len(inds)+1)]))
    
    counter=1;
    curid=-1;
    for (p,qid) in izip(rels,qs):
        if curid>0 and curid!=qid:
            output.append(printtable(table));
            table={};
            counter=1;
        table[counter]=float(p);    
        curid=qid;
        counter+=1;
    output.append(printtable(table))
    return(output)



def rel2labels(rels,qs):
    output=[]
    curid=''
    table={}

    def printtable(table):
        inds=table.items();
        inds.sort(key=operator.itemgetter(0));
        return(' '.join(map(lambda (a,b): str(b),inds)))
    
    counter=1;
    curid=-1;
    for (p,qid) in izip(rels,qs):
        if curid>0 and curid!=qid:
            output.append(printtable(table));
            table={};
            counter=1;
        table[counter]=float(p);    
        curid=qid;
        counter+=1;
    output.append(printtable(table))
    return(output)



def getrmse(preds,labels):
	mse=0.0;
        for (pr,ta) in izip(preds,labels):
	    mse+=(pr-ta)**2;
	return(sqrt(mse/float(len(preds))))


def evaluate(preds,queries,labels,max_grade=2):
    rmse=getrmse(preds,labels);
    #print rmse
    
    ranks=rel2ranks(preds,queries);
    #print ranks;
    labs=rel2labels(labels,queries);
    #print labs;
    #print len(ranks),len(labs);
    assert(len(ranks)==len(labs));

    [err,ndcg]=evaluate_submission(labs,ranks,max_grade=max_grade);
    return([rmse,err,ndcg])


def bestalpha(preds,deltapreds,queries,labels,alphas):
    #print rmse
    
    labs=rel2labels(labels,queries);

    bestERR=-1.0;
    bestA=-1;
    for a in alphas:        
        ranks=rel2ranks([preds[i]+a*deltapreds[i] for i in range(0,len(preds))],queries);
        assert(len(ranks)==len(labs));
        [err,ndcg]=evaluate_submission(labs,ranks);
        if err>bestERR:
            bestERR=err;
            bestA=a;
            bestRMSE=getrmse(preds,labels)
            bestNDCG=ndcg
            
    return [bestRMSE,bestERR,bestNDCG,bestA]
