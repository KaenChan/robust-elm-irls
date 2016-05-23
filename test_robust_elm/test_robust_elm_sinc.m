function [testEVAL_rbelm, testEVAL_elm]=test_robust_elm_sinc(option)
%dbstop if error
option.c_rho = 27;
option.Nh_nodes = 1000;
if ~isfield(option, 'loss_type')
    option.loss_type = 'bisquare';
    option.regu_type = 'l2';
    % option.regu_type = 'none';
    option.inv_type = 'svd';
    option.noise = 0.4;
    option.data_iter = 2;
end
% filename = ['data/data_sinc/data_sinc_1000_' num2str(option.noise) '_' num2str(option.data_iter) '.mat'];
% filename
option.stop_delta = 0.1;
option.seed = 0;
option.Max_iters = 20;

% cv params for L=100
option.Nh_nodes = 100;
option.c_rho = 27;
if strcmp(option.regu_type,'l1')
    if strcmp(option.loss_type,'l1') || strcmp(option.loss_type,'bisquare')
        option.c_rho = 20;
    else
        option.c_rho = 18;
    end
end
if option.noise==0
    option.c_rho = 28;
end

if strcmp(option.loss_type,'l1')
    option.metric_type.name = 'mae';
    option.tune = 1;
elseif strcmp(option.loss_type,'huber')
    option.metric_type.name = 'mae';
    option.alpha = 0.90;
    option.tune = 1.345;
    % option.tune = 0.9;
    if option.noise==0.3
        option.tune = option.tune * 0.1; % for 30% outlier
    elseif option.noise==0.4
        option.tune = option.tune * 0.1; % for 40% outlier
    end
elseif strcmp(option.loss_type,'bisquare')
    option.metric_type.name = 'mae';
    option.alpha = 0.95;
    option.tune = 4.685;
    if option.noise==0.3
        option.tune = option.tune * 0.7; % for 30% outlier
    elseif option.noise==0.4
        option.tune = option.tune * 0.3; % for 40% outlier
    end
    % option.tune = 6;
    % option.tune = 4;
elseif strcmp(option.loss_type,'cauchy')
    option.metric_type.name = 'mae';
    option.alpha = 0.90;
    option.tune = 2.385;
elseif strcmp(option.loss_type,'welsch')
    option.metric_type.name = 'mae';
    option.alpha = 0.90;
    option.tune = 2.985;
    if option.noise==0.3
        option.tune = option.tune * 0.3; % for 30% outlier
    elseif option.noise==0.4
        option.tune = option.tune * 0.3; % for 40% outlier
    end
end
option.metric_type.name = 'rmse';

% filename='data_sinc_1000_0_1';
% if isfield(option, 'filename')
%     filename = option.filename;
% end
[X_train, Y_train, X_vali, Y_vali, X_test, Y_test] = gen_sinc_data(0.2, 200);
% scatter(X_train, Y_train);
% save data_sinc_1000_0_10 X_train  Y_train X_vali Y_vali X_test Y_test;
% load data_sinc_1000_10_5;
% load(filename);
% load data_sinc;
% load data_sinc_noise10_3;
% load('E:\work\machine_learning-ref\ELM\workspace\src\gb-elm\data\sinc\data_sinc_noise10_3.mat')
% load data_sinc_noise20;
% load data_sinc_noise40;
% 
% d_mean = mean(X_train);
% d_std = std(X_train);
% d_mean = min(X_train);
% d_std = max(X_train) - min(X_train);
% X_train = (X_train - d_mean) ./ d_std;
% X_vali = (X_vali - d_mean) ./ d_std;
% X_test = (X_test - d_mean) ./ d_std;

dataset = 'sinc';

info = '';
info = [info sprintf('dataset      = %s\n', dataset)];
info = [info sprintf('trainsize    = %s\n', mat2str(size(X_train)))];
info = [info sprintf('testsize     = %s\n', mat2str(size(X_test)))];
info = [info sprintf('loss_type    = %s\n', option.loss_type)];
info = [info sprintf('regu_type    = %s\n', option.regu_type)];
info = [info sprintf('metric_type  = %s\n', option.metric_type.name)];
info = [info sprintf('Nh_nodes     = %s\n', mat2str(option.Nh_nodes))];
info = [info sprintf('Max_iters    = %d\n', option.Max_iters)];
info = [info sprintf('c_rho-elm    = %s\n', mat2str(option.c_rho))];
info = [info sprintf('tune         = %s\n', mat2str(option.tune))];
info = [info sprintf('\n')];

fprintf(info);

option.elm_type           = 'regression';
option.act_func = 'sig';
option.X_test             = X_test;
option.Y_test             = Y_test;
option.plot               = 2;
option.verbose            = 1;

warning('on');
model = robust_elm_train(X_train, Y_train, option);
warning('on');
fprintf('Num of less than 1e-10: %d\n',sum(model.OutputWeight<1e-10));

pred_train = elm_predict(model, X_train, Y_train);
pred_vali= elm_predict(model, X_vali, Y_vali);
pred_test= elm_predict(model, X_test, Y_test);
TrainEVAL = compute_metric(pred_train, Y_train, [], option.metric_type);
ValidEVAL = compute_metric(pred_vali, Y_vali, [], option.metric_type);
TestEVAL  = compute_metric(pred_test, Y_test, [], option.metric_type);
model.EVAL = [TrainEVAL ValidEVAL TestEVAL];
testEVAL_rbelm = TestEVAL;
option.plot = 0;
if option.plot
    % close all;
    figure(2); % clf;
    % scatter(X_train, Y_train, 30);
    scatter(X_train, Y_train, 30, 5-Y_train+pred_train)
    hold on;
    plot(X_train, pred_train, 'r', 'LineWidth',3)
    drawnow;
end
fprintf('\n');
fprintf('ELM-IRLS N=%-8d | TrainTime=%.4f s | %s (%.4f %.4f %.4f) ||\n', ...
    model.N, model.TrainTime, option.metric_type.name, model.EVAL(1), model.EVAL(2), model.EVAL(3));

option.c_rho = 27;
if option.noise==0
    option.c_rho = 28;
end
model = run_basic_elm(X_train, Y_train, X_vali, Y_vali, X_test, Y_test, option);
testEVAL_elm = model.EVAL(3);


function model = run_basic_elm(X_train, Y_train, X_vali, Y_vali, X_test, Y_test, option)
option.n_hidden_nodes = option.Nh_nodes;
option.act_func    = 'sig';
option.c_rho       = option.c_rho;
option.metric_type.k_ndcg = 0;
model = elm_train(X_train, Y_train, option);
model.N = option.n_hidden_nodes;
model.EVAL = [0, 0, 0];

pred_train = elm_predict(model, X_train, Y_train);
pred_vali= elm_predict(model, X_vali, Y_vali);
pred_test= elm_predict(model, X_test, Y_test);
TrainEVAL = compute_metric(pred_train, Y_train, [], option.metric_type);
ValidEVAL = compute_metric(pred_vali, Y_vali, [], option.metric_type);
TestEVAL  = compute_metric(pred_test, Y_test, [], option.metric_type);
model.EVAL = [TrainEVAL ValidEVAL TestEVAL];

if option.plot
    figure(2);
    % scatter(X_train, Y_train, 20);
    hold on;
    plot(X_train, pred_train, 'k--', 'LineWidth', 3)
    drawnow;

    % save('result/sinc_elm_l2', 'X_train', 'Y_train', 'pred_train');
end

fprintf('\n');
fprintf('ELM      N=%-8d | TrainTime=%.4f s | %s (%.4f %.4f %.4f) ||\n', ...
    model.N, model.TrainTime, ...
option.metric_type.name, model.EVAL(1), model.EVAL(2), model.EVAL(3));
