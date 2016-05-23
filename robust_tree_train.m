function model = robust_tree_train(X_train, Y_train, option)

%% input option
metric_type        = option.metric_type;
loss_type          = option.loss_type;
Max_iters          = option.Max_iters;

if isfield(option, 'tune')
    tune = option.tune;
end

if isfield(option, 'tune_relative')
    if strcmp(option.loss_type,'l1')
        default_tune = 1;
    elseif strcmp(option.loss_type,'huber')
        default_tune = 1.345;
    elseif strcmp(option.loss_type,'bisquare')
        default_tune = 4.685;
    elseif strcmp(option.loss_type,'cauchy')
        default_tune = 2.385;
    elseif strcmp(option.loss_type,'welsch')
        default_tune = 2.985;
    end
    tune = option.tune_relative * default_tune;
end
    

X_train = full(X_train);
Y_train = full(Y_train);
if isfield(option, 'X_test')
    X_test = option.X_test;
    Y_test = option.Y_test;
    X_test = full(X_test);
    Y_test = full(Y_test);
else
    X_test=[];Y_test=[];
end
option.rank_type = 'pointwise';

if isfield(option, 'seed')
    seed = option.seed;
else
    seed = fix(mod(cputime,100));
end
rand('seed',seed);
% rng(0);

loss.train.e = [];
loss.test.e  = [];
T=Y_train;
T=double(T);
t1=clock;

%%
K = size(Y_train,2);

tic;
tiny_s = 1e-6 * std(Y_train);
if tiny_s==0
    tiny_s = 1;
end

tic;
regressor = fitrtree(X_train, Y_train, ...
    'MaxNumSplits', option.MaxNumSplits, 'Prune','off');
Y = predict(regressor, X_train);

if ~isempty(Y_test)
    pred_test = predict(regressor, X_test);
end

train_loss = 1;
TrainEVAL = compute_metric(Y, Y_train, [], metric_type);
TestEVAL = 0;
if ~isempty(Y_test)
    TestEVAL  = compute_metric(pred_test, Y_test, [], metric_type);
end
consumed_time = toc;
if option.verbose == 1
    fprintf('%.2f s (%.2f s) | ---------rls-elm----------- | %s %.4f - %.4f ', ...
        consumed_time, consumed_time, metric_type.name, TrainEVAL, TestEVAL);
    fprintf(' |\n');
end
% for iter=1:20
%     diff = H*out_w - T;
%     diff = diff/(max(tune,tiny_s)*tune);
%     W = robust_func(diff, 'l1', 'wgt');
%     W = sparse(W);
%     W = diag(W);
%     out_w = (speye(size(H,2))/c_rho + H'*W*H) \ (H'*(W*T));
% end
diff = Y - T;
s = madsigma(diff,1);
diff = diff/(max(s,tiny_s)*tune);
% Max_iters = 100;
D = sqrt(eps(class(X_train)));
for iter=1:Max_iters
    %%
    switch lower(loss_type)
        case 'l1'
            diff = Y - T;
            s = madsigma(diff,1);
            diff = diff/(max(s,tiny_s)*tune);
            % delta = 0;
            W = robust_func(diff, loss_type, 'wgt');
        case {'huber','bisquare','cauchy', 'welsch'}
            diff = Y - T;
            s = madsigma(diff,1);
            diff = diff/(max(s,tiny_s)*tune);
            % delta = quantile(abs(diff), option.alpha);
            W = robust_func(diff, loss_type, 'wgt', 1);
        otherwise
            warning('error');
    end

    regressor = fitrtree(X_train, Y_train, 'Weights', W, ...
        'MaxNumSplits', option.MaxNumSplits, 'Prune','off');
    Y = predict(regressor, X_train);

    %% 
    pred_train = Y;
    if ~isempty(Y_test)
        pred_test = predict(regressor, X_test);
    end
    valid_interval = 1;
    train_loss = mean(robust_func(diff, loss_type, 'rho', 1));
    if mod(iter, valid_interval)==0
        consumed_time = toc;
        TrainEVAL = compute_metric(pred_train, Y_train, [], metric_type);
        TestEVAL = 0;
        if ~isempty(Y_test)
            TestEVAL  = compute_metric(pred_test, Y_test, [], metric_type);
        end
        t2=clock;
        TrainingTime=etime(t2,t1);
        if option.verbose == 1
            fprintf('%.2f s (%.2f s) | iter %d | %s loss: %.4f | %s %.4f - %.4f ', ...
                TrainingTime, consumed_time, iter, loss_type, train_loss, metric_type.name, TrainEVAL, TestEVAL);
            fprintf(' |\n');
        end
        loss.train.e(end+1) = TrainEVAL;
        loss.test.e(end+1) = TestEVAL;
        if option.plot == 1
            figure(1) ; clf ;
            ki = iter/valid_interval;
            plot((1:ki)*valid_interval, loss.train.e, 'k') ; hold on ;
            plot((1:ki)*valid_interval, loss.test.e, 'r') ;
            h=legend('train', 'test') ;
            grid on ;
            xlabel('Num of iteration') ; ylabel(metric_type.name) ;
            set(h,'color','none') ;
            title(metric_type.name) ;
            drawnow;
        elseif option.plot == 2
            figure(2);  clf;
            scatter(X_train, Y_train, 30, 5-Y_train+pred_train)
            hold on;
            plot(X_train, pred_train, 'r','LineWidth',3)
            drawnow;
        end

        tic;
    end
end

t2 = clock;
TrainingTime = etime(t2,t1);

model.iter = iter;
model.regressor = regressor;
model.MaxNumSplits = option.MaxNumSplits;
model.EVAL = [TrainEVAL TestEVAL];
model.metric_type = metric_type;
model.TrainTime = TrainingTime;
model.loss = loss;

% -----------------------------
function s = madsigma(r,p)
%MADSIGMA    Compute sigma estimate using MAD of residuals from 0
rs = sort(abs(r));
s = median(rs(max(1,p):end)) / 0.6745;
