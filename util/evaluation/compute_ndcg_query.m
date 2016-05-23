function ndcg_1q = compute_ndcg_query(r)
    dcg = compute_dcg_query(r);
    ideal_dcg = compute_dcg_query(sort(r, 'descend'));
    if ideal_dcg > 0
        ndcg_1q = dcg ./ ideal_dcg;
    % elseif dcg == ideal_dcg         % kaggle的实现方法https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain
    %     ndcg_1q = 1;
    else
        ndcg_1q = zeros(1,length(r));
    end;
