robust-elm-irls
===========================

## Citation
If you use the codes, please cite the following paper:
```
@article{Chen2016Robust,
  title={Robust regularized extreme learning machine for regression using iteratively reweighted least squares},
  author={Chen, Kai and Lv, Qi and Lu, Yao and Dou, Yong},
  journal={Neurocomputing},
  volume={230C},
  pages={489--501},
  year={2017},
  publisher={Elsevier}
}
```
    
## Overview
Robust-elm-irls is the robust regularized Extreme Learning Machine for regression using Iteratively Reweighted Least Squares (IRLS).

Robust loss function:
- L1-norm loss function
- Huber loss function
- Bisquare loss function
- Welsch loss function

Regularization:
- L2-norm regularization
- L1-norm regularization

## Demo


```
>> addpath(genpath('.'))
>> test_robust_elm_sinc.m
```
<img src="https://github.com/KaenChan/robust-elm-irls/blob/master/test_robust_elm/sinc_result.jpg" height="300" width="400" >


```
>> addpath(genpath('.'))
>> test_robust_elm_run.m
dataset      = mpg
trainsize    = [235 7]
testsize     = [157 7]
loss_type    = bisquare
regu_type    = l2
metric_type  = rmse
Nh_nodes     = 200
c_rho-elm    = 4

0.02 s (0.02 s) | ---------rls-elm----------- | rmse 8.4299 - 4.4893  |
0.04 s (0.03 s) | iter 1 | bisquare loss: 383.5101 | rmse 8.9299 - 3.0993  |
0.06 s (0.02 s) | iter 2 | bisquare loss: 185.8657 | rmse 9.4228 - 2.7330  |
0.08 s (0.02 s) | iter 3 | bisquare loss: 120.8517 | rmse 9.6767 - 2.7376  |
0.10 s (0.02 s) | iter 4 | bisquare loss: 104.9857 | rmse 9.7019 - 2.7499  |
0.12 s (0.02 s) | iter 5 | bisquare loss: 104.4186 | rmse 9.7059 - 2.7549  |
0.14 s (0.01 s) | iter 6 | bisquare loss: 104.4091 | rmse 9.7071 - 2.7574  |
0.16 s (0.02 s) | iter 7 | bisquare loss: 104.1917 | rmse 9.7075 - 2.7587  |
0.17 s (0.02 s) | iter 8 | bisquare loss: 104.0996 | rmse 9.7077 - 2.7594  |
0.19 s (0.02 s) | iter 9 | bisquare loss: 104.0486 | rmse 9.7078 - 2.7598  |
0.20 s (0.02 s) | iter 10 | bisquare loss: 104.0212 | rmse 9.7079 - 2.7600  |
0.22 s (0.02 s) | iter 11 | bisquare loss: 104.0063 | rmse 9.7079 - 2.7601  |
0.24 s (0.02 s) | iter 12 | bisquare loss: 103.9982 | rmse 9.7079 - 2.7601  |
0.26 s (0.02 s) | iter 13 | bisquare loss: 103.9938 | rmse 9.7079 - 2.7602  |
0.28 s (0.02 s) | iter 14 | bisquare loss: 103.9915 | rmse 9.7079 - 2.7602  |
0.30 s (0.02 s) | iter 15 | bisquare loss: 103.9902 | rmse 9.7079 - 2.7602  |
0.32 s (0.02 s) | iter 16 | bisquare loss: 103.9895 | rmse 9.7079 - 2.7602  |
0.34 s (0.02 s) | iter 17 | bisquare loss: 103.9891 | rmse 9.7079 - 2.7602  |
0.35 s (0.02 s) | iter 18 | bisquare loss: 103.9889 | rmse 9.7079 - 2.7602  |
0.37 s (0.02 s) | iter 19 | bisquare loss: 103.9888 | rmse 9.7079 - 2.7602  |
0.39 s (0.02 s) | iter 20 | bisquare loss: 103.9888 | rmse 9.7079 - 2.7602  |
Num of less than 1e-10: 76
TrainTime=0.3870 s | rmse (9.7079 2.7602) ||
```
