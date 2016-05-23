function score = compute_r2score(Ypred,Yt)
    % """R^2 (coefficient of determination) regression score function.
    % Best possible score is 1.0, lower values are worse.
    % 
    % References
    % ----------
    % .. [1] `Wikipedia entry on the Coefficient of determination
    %         <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_

	y_true=Yt{1};
	for i=2:length(Yt)
	    y_true = [y_true; Yt{i}];
    end
	y_pred = Ypred;

	res = sum((y_true - y_pred) .^ 2);
	tot = sum((y_true - mean(y_true)) .^ 2);

	if tot == 0.0
	    if res == 0.0
	        score = 1.0;
        else
	        % arbitrary set to zero to avoid -inf scores, having a constant
	        % y_true is not interesting for scoring a regression anyway
	        score = 0;
        end
    end;

	score = 1 - res / tot;