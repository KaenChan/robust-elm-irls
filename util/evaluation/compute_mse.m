function [mse, mse_arr] = compute_mse(Ypred,Yt,qids)
% mean_squred_error
  ind = 0;
  if isempty(qids)
      qids = {1:length(Yt)};
  end
  for i=1:length(qids)
  	pred = Ypred(qids{i});
    y = Yt(qids{i});
    mse_arr(i) = mean((pred-y).^2);
  end;
  mse_arr = mse_arr';
  mse=mean(mse_arr);