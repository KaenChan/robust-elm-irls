function [ndcg, ndcg_arr] = compute_ndcg(Ypred, Yt, qids, k)
    % if k==0 return MeanNDCG
    ind = 0;
    ndcg = 0;
    for i=1:length(qids)
        [foo,ind2] = sort(-Ypred(qids{i}));
	    y = Yt(qids{i});
        r = y(ind2);
        ndcg_temp = compute_ndcg_query(r);
        if k>length(r)
            idx=r;
        else
            idx=k;
        end
        ndcg_temp = compute_ndcg_query(r);
        if k==0
            ndcg_arr(i) = mean(ndcg_temp);
        else
            ndcg_arr(i) = ndcg_temp(idx);
        end
    end;
    ndcg_arr = ndcg_arr';
    ndcg=mean(ndcg_arr);
