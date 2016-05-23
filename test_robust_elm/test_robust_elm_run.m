function test_robust_elm_run(idxdata, loss_type, regu_type)
% clear;

if nargin==0
    idxdata=3;
    loss_type = 'bisquare';
    regu_type = 'l2';
end

noise_in=[0 0.1 0.2 0.3 0.4];
noise_in=0.3;

datasets{1} = 'space_ga';
datasets{2} = 'cadata';
datasets{3} = 'mpg';
datasets{4} = 'mg';
datasets{5} = 'cpusmall';
datasets{6} = 'data_sinc_1000';

dataset = datasets{idxdata};
iter=1;
t1=tic;
for noise=noise_in
    
    [folder, name, ext] = fileparts(which('test_robust_elm_grid'));
    modelfile = [folder '/models/' dataset '/robustelm_' dataset ...
        '_' loss_type '_' regu_type ...
        '_noise' num2str(noise) ...
        '.json']
    option = loadjson(modelfile);
    
    for i=3
        noise
        pathname = ['test_robust_elm/data/' dataset];
        filename = [pathname '/' dataset '_' num2str(noise) '_' num2str(i) '.mat']
        load(filename);

        info = '';
        info = [info sprintf('dataset      = %s\n', dataset)];
        info = [info sprintf('trainsize    = %s\n', mat2str(size(X_train)))];
        info = [info sprintf('testsize     = %s\n', mat2str(size(X_test)))];
        info = [info sprintf('loss_type    = %s\n', option.loss_type)];
        info = [info sprintf('regu_type    = %s\n', option.regu_type)];
        info = [info sprintf('metric_type  = %s\n', option.metric_type.name)];
        info = [info sprintf('Nh_nodes     = %s\n', mat2str(option.Nh_nodes))];
        info = [info sprintf('c_rho-elm    = %s\n', mat2str(option.c_rho))];
        info = [info sprintf('\n')];
        fprintf(info);
    
        option.X_test             = X_test;
        option.Y_test             = Y_test;
        option.plot               = 0;
        option.verbose            = 1;
        option.metric_type.name = 'rmse';
        option.inv_type = 'svd';
        % option.Max_iters = 50;

        ttrain=tic;
        warning('off');
        model = robust_elm_train(X_train, Y_train, option);
        warning('on');
        TrainTime = toc(ttrain);
        fprintf('Num of less than 1e-10: %d\n',sum(model.OutputWeight<1e-10));
        pred_train = elm_predict(model, X_train, Y_train);
        pred_test = elm_predict(model, X_test, Y_test);
        TrainEVAL = compute_metric(pred_train, Y_train, [], option.metric_type);
        TestEVAL  = compute_metric(pred_test, Y_test, [], option.metric_type);
        model.EVAL = [TrainEVAL TestEVAL];
        fprintf('TrainTime=%.4f s | %s (%.4f %.4f) ||\n', ...
            model.TrainTime, option.metric_type.name, model.EVAL(1), model.EVAL(2));
        TestEVALs(iter, i) = TestEVAL;
        TrainTimes(iter, i) = TrainTime;
        sparserate(iter, i) = sum(model.OutputWeight < 1e-10) / length(model.OutputWeight);
    end
    result.n_hiddens(iter) = length(model.OutputWeight);
    result.c_rho(iter) = option.c_rho;
    result.tune_relative(iter) = option.tune_relative;
    iter = iter+1;
end

clear result;
result.dataset = dataset;
result.trainsize = size(X_train);
result.testsize = size(X_test);
result.TestEVALs = TestEVALs;
result.TrainTimes = TrainTimes;
result.sparserate = sparserate ;
result.MeanTest = mean(TestEVALs, 2);
result.StdTest = std(TestEVALs,[],2);
result.MeanTrainTime = mean(mean(TrainTimes));
result.StdTrainTime = std(reshape(TrainTimes,[],1));
s = '';
for i=1:length(result.MeanTest)
    s = [s sprintf('%.4f$\\pm$%.3f & ',result.MeanTest(i),result.StdTest(i))];
end
result.tex = s;

[folder, name, ext] = fileparts(which('test_robust_elm_real_grid'));
resultfilename = [folder '/results/robustelm_' dataset ...
    '_' loss_type '_' regu_type '.json']
% resultfilename = [folder '/results/robustelm_' dataset '_noise' num2str(noise)...
%     '_' loss_type '_' regu_type '.json']
% resultfilename='1.json';
resultfilename = ['robustelm_' dataset '_' loss_type '_' regu_type '.json'];
savejson('',result,resultfilename);
result.tex
% mean(result.n_hiddens)
% mean(result.sparserate)
toc(t1)
