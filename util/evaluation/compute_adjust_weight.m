function [tau, mu] = compute_adjust_weight(Ypred,Yt,k)
    ind = 0;
    ndcg = 0;
    n_querys = length(Yt);
    max_grade = max(max(Yt))+1;
    tau = zeros(max_grade*(max_grade-1)/2, 1);
    for i=1:n_querys
        ind = ind(end)+[1:length(Yt{i})];
        [foo,ind2] = sort(-Ypred(ind));
        r = Yt{i}(ind2);
        tau = tau + compute_tau_1q(r, max_grade);
    end;
    tau=tau/length(Yt);
    
    mu = zeros(n_querys,1);

function tau = compute_tau_1q(r, max_grade)
    r = sort(r, 'descend');
    ideal_ndcg = compute_ndcg_1q(r,1);
    tau = zeros(max_grade*(max_grade-1)/2, 1);
    grades_here = unique(r);
    if length(grades_here) > 1
        for i = 2:length(grades_here)
            for j= 1:i
                a = grades_here(i);
                b = grades_here(j);
                drop_ndcg = 0;
                n_rand = 20;
                for k = 1:n_rand    %随机多次
                    a_idx = find(r==a);
                    a_rand_idx = int32(length(a_idx)*rand(1));
                    b_idx = find(r==b);
                    b_rand_idx = int32(length(b_idx)*rand(1));
                    r_temp = r;
                    r_temp(a_rand_idx) = r(b_rand_idx);
                    r_temp(b_rand_idx) = r(a_rand_idx);
                    ndcg = compute_ndcg_1q(r_temp, 1);
                    drop_ndcg = drop_ndcg + ideal_ndcg - ndcg;
                end
                drop_ndcg = drop_ndcg / n_rand;
                tau_i = sum(0:i-2)+j-1;
                tau(tau_i) = drop_ndcg;
            end
        end
    end

function ndcg_1q = compute_ndcg_1q(r,k)
    dcg = compute_dcg(r,k);
    ideal_dcg = compute_dcg(sort(r, 'descend'),k);
    if ideal_dcg > 0
        ndcg_1q = dcg / ideal_dcg;
    elseif dcg == ideal_dcg         % kaggle的实现方法https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain
        ndcg_1q = 1;
    else
        ndcg_1q = 0;
    end;

function dcg = compute_dcg(r,k)
    if k<=size(r,1)
        r = r(1:k);
    else
        r = [r', zeros(1, k-size(r,1))]';
    end;
    g = power(2, r) - 1;
    dg = g./log2([2:k+1]');
    dcg = sum(dg);
    