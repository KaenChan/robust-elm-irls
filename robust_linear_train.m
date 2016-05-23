function model = robust_linear_train(X_train, Y_train, option)

%% input option
metric_type        = option.metric_type;
c_rho              = 2^option.c_rho;
loss_type          = option.loss_type;
Max_iters          = option.Max_iters;
tune = option.tune;

X_test = option.X_test;
Y_test = option.Y_test;

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

H = [X_train ones(size(X_train,1),1)];
H_test = [X_test ones(size(X_test,1),1)];

tic;
out_w = (speye(size(H,2))/c_rho + H'*H) \ (H'*T);
Y = H*out_w;
out_w_pre = out_w;

pred_test = H_test * out_w;

train_loss = 1;
TrainEVAL = compute_metric(Y, Y_train, [], metric_type);
TestEVAL  = compute_metric(pred_test, Y_test, [], metric_type);
consumed_time = toc;
if option.verbose == 1
    fprintf('%.2f s (%.2f s) | ---------irls-elm----------- | %s %.4f - %.4f ', ...
        consumed_time, consumed_time, metric_type.name, TrainEVAL, TestEVAL);
    fprintf(' |\n');
end

tiny_s = 1e-6 * std(Y_train);
if tiny_s==0
    tiny_s = 1;
end

% Max_iters = 100;
D = sqrt(eps(class(X_train)));
for iter=1:Max_iters
    if((iter~=1) && any(abs(out_w-out_w_pre) <= D*max(abs(out_w),abs(out_w_pre))))
        break
    end
    out_w_pre = out_w;
    %%
    switch lower(loss_type)
        case 'l1'
            diff = Y - T;
            delta = 0;
            W = robust_func(diff, loss_type, 'wgt');
        case {'huber','bisquare','cauchy', 'welsch'}
            diff = Y - T;
            s = madsigma(diff,1);
            diff = diff/(max(s,tiny_s)*tune);
            delta = quantile(abs(diff), option.alpha);
            W = robust_func(diff, loss_type, 'wgt', 1);
        otherwise
            warning('error');
    end
    W = sparse(W);
    W = diag(W);

    out_w = (speye(size(H,2))/c_rho + H'*W*H) \ (H'*(W*T));

    Y = H*out_w;

    %% 
    pred_train = Y;
    pred_test = H_test * out_w;

    valid_interval = 1;
    train_loss = sum(robust_func(T-pred_train, loss_type, 'rho', 1));
    if mod(iter, valid_interval)==0
        consumed_time = toc;
        TrainEVAL = compute_metric(pred_train, Y_train, [], metric_type);
        TestEVAL  = compute_metric(pred_test, Y_test, [], metric_type);
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
            ki = (k - 1)*Max_iters+iter/valid_interval;
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
            scatter(X_train, Y_train, 'b');
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
model.OutputWeight = out_w;
model.EVAL = [TrainEVAL TestEVAL];
model.metric_type = metric_type;
model.TrainTime = TrainingTime;
model.loss = loss;
model.c_rho = option.c_rho;

%% -----------------------------
function s = madsigma(r,p)
%MADSIGMA    Compute sigma estimate using MAD of residuals from 0
rs = sort(abs(r));
s = median(rs(max(1,p):end)) / 0.6745;
