function dcg = compute_dcg_query(r)
    % g = power(2, r) - 1;
    % dg = g./log2([2:k+1]');
    % dcg = sum(dg);

    % letor
    k=length(r);
    g(1) = 2^r(1) - 1;
    g(2) = 2^r(2) - 1;
    g(3:k) = (power(2, r(3:k)) - 1) ./ log2([3:k]');
    % dcg = sum(g);
    dcg = cumsum(g);
