function [lad, lad_arr] = compute_mae(Ypred,Yt,qids)
% mean_absolute_error
  ind = 0;
  if isempty(qids)
      qids = {1:length(Yt)};
  end
  for i=1:length(qids)
  	pred = Ypred(qids{i});
    y = Yt(qids{i});
    lad_arr(i) = mean(abs(pred-y));
  end;
  lad_arr = lad_arr';
  lad=mean(lad_arr);