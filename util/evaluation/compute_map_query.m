function map = compute_map_query(r)
  r = r>0;
  p = cumsum(r) ./ [1:length(r)]';
  if sum(r)> 0 
    map = r'*p / sum(r);
  else
    map = 0;
  end;
