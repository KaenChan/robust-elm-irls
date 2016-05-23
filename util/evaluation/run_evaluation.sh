#!bash

dataset=E:/work/ml-datasets/letor3.0/OHSUMED/QueryLevelNorm

rm result.txt

for i in 1 2 3 4 5
do
    echo fold$i
    perl E:/work/ml-datasets/letor4.0/Eval-Score-4.0.pl ${dataset}/Fold${i}/test.txt ${dataset}/rankelm/test.fold${i} temp 0
    cat temp
    cat temp >> result.txt

done


for i in 1 2 3 4 5
do
    echo fold$i
    python e:/work/ml-work/learning-to-rank/src-work/pgbrt/pgbrt/scripts/evaluate.py ${dataset}/Fold${i}/test.txt ${dataset}/rankelm/test.fold${i}
done



