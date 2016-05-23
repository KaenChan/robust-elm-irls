function [map, map_arr] = compute_map(Ypred, Yt, qids)
  ind = 0;
  maxr = max(Yt);
  if maxr==4
    mtype='mslr'; 
  else
    mtype='letor'; 
  end;
  for i=1:length(qids)
    [foo,ind2] = sort(-Ypred(qids{i}));
    y = Yt(qids{i});
    if strcmp(mtype,'letor')
        r = y(ind2)>0;
    else
        r = y(ind2)>1;
    end
    p = cumsum(r) ./ [1:length(r)]';
    if sum(r)> 0 
      map_arr(i) = r'*p / sum(r);
    else
      map_arr(i)=0;
    end;
  end;
  map_arr = map_arr';
  map=mean(map_arr);