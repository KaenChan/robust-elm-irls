function auc = compute_auc(pred,Yt,posclass)
% compute auc for bipartite rank
    labels = Yt==posclass;
    scores = (pred-min(pred))/(max(pred)-min(pred));
    auc = fastAUC(labels,scores);
%     [~,~,~,auc] = perfcurve(Yt,scores,posclass);
