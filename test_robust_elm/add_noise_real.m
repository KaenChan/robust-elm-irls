function y = add_noise_real(y, n_outliers)
if n_outliers==0
    return
end
if n_outliers<=1
    n_outliers = fix(length(y)*n_outliers);
end

% y_m = quantile(abs(y), 0.5);
% expect_y = mean(abs(y-y_m));
% y_noise_1 = random('Normal', 0, 1, length(y), 1);
% scale = expect_y / mean(abs(y_noise_1));
% y_noise_1 = scale * y_noise_1 * 0.5;
% y = y + y_noise_1;

ymin = min(y);
ymax = max(y);
noises = rand(n_outliers, 1)*(ymax-ymin)+ymin;
idxs = randi(length(y), n_outliers, 1);
y(idxs) = noises*1.2;
