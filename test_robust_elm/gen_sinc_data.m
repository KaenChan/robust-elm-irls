function [X_train, Y_train, X_vali, Y_vali, X_test, Y_test] = gen_sinc_data(ratio_noise, num)
% ratio_noise = 0.1;

% seed = fix(mod(cputime,100));
% rand('seed',seed);
x = linspace(-4,4,num);
x = x';
y = sinc(x);
y = add_gaussian_noise(y);
y = add_bern_noise(y, ratio_noise);
X_train = x;
Y_train = y;
% figure(1)
% scatter(x,y,'r')

x = linspace(-4,4,num);
x = x';
y = sinc(x);
y = add_gaussian_noise(y);
y = add_bern_noise(y, ratio_noise);
X_vali = x;
Y_vali = y;
% figure(2)
% scatter(x,y,'b')

x = linspace(-4,4,num);
x = x';
y = sinc(x);
y = add_gaussian_noise(y);
X_test = x;
Y_test = y;
% figure(3)
% scatter(x,y,'g')

function y = add_constant_noise(y, n_outliers)
    if n_outliers<=1
        n_outliers = fix(length(x)*n_outliers);
    end
    noises = [-0.5 0.8 1]';
    noises = [-1 1]';
    y_noise = randi(length(noises), n_outliers, 1);
    y_noise = noises(y_noise);
    idxs = randi(length(x), n_outliers, 1);
    y(idxs) = y(idxs) + y_noise;

function y = add_gaussian_noise(y)
    y_noise = random('Normal', 0, 0.1, length(y), 1);
    y = y + y_noise;

function y = add_bern_noise(y, p)
    y_noise = BernProc(length(y), p);
    y = y + y_noise;

function bern = BernProc(n,p)
     % bern = -1+2*(rand(n,1) <= p);
     bern = (rand(n,1) <= p);
