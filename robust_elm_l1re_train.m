function model = robust_elm_l1re_train(X_train, Y_train, option)

%% input option
Nh_nodes           = option.Nh_nodes;
act_func = option.act_func;
metric_type        = option.metric_type;
c_rho              = 2^option.c_rho;
loss_type          = option.loss_type;
elm_type           = option.elm_type;
Max_iters          = option.Max_iters;
tune = option.tune;
X_test = option.X_test;
Y_test = option.Y_test;
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

[H, iw, bias] = elm_hidden_layer_gen(X_train, Nh_nodes, act_func);
H_test = elm_hidden_layer_apply(X_test, iw, bias, act_func);
tic;
out_w = (speye(size(H,2))/c_rho + H'*H) \ (H'*T);
% for iter=1:20
%     diff = H*out_w - T;
%     diff = diff/(max(tune,tiny_s)*tune);
%     W = robust_func(diff, 'l1', 'wgt');
%     W = sparse(W);
%     W = diag(W);
%     out_w = (speye(size(H,2))/c_rho + H'*W*H) \ (H'*(W*T));
% end
% out_w = zeros(Nh_nodes, size(T,2));
Y = H*out_w;
out_w_pre = out_w;

pred_test = H_test * out_w;

train_loss = 1;
TrainEVAL = compute_metric(Y, Y_train, [], metric_type);
TestEVAL  = compute_metric(pred_test, Y_test, [], metric_type);
consumed_time = toc;
if option.verbose == 1
    fprintf('%.2f s (%.2f s) | ---------rls-elm----------- | %s %.4f - %.4f ', ...
        consumed_time, consumed_time, metric_type.name, TrainEVAL, TestEVAL);
    fprintf(' |\n');
end
%% Max_iters = 100;
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
            diff = diff/(max(tune,tiny_s)*tune);
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
    
    z = out_w;
    s = madsigma(z,1);
    z = z/(max(s,tiny_s)*tune);
    W_beta = 1 ./ max(abs(z),0.000001);
    W_beta = diag(sparse(W_beta));

    out_w = (W_beta/c_rho + H'*W*H) \ (H'*(W*T));
    
    Y = H*out_w;

    %% 
    pred_train = Y;
    pred_test = H_test * out_w;

    valid_interval = 1;
    train_loss = mean(robust_func(T-pred_train, loss_type, 'rho', delta));
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

%%
t2 = clock;
TrainingTime = etime(t2,t1);

if size(H,2)<=200
t1=tic;
brob = robustfit(H, Y_train, option.loss_type);
t2=toc(t1);
pred_train = [ones(size(H,1),1) H] * brob;
pred_test =  [ones(size(H_test,1),1) H_test] * brob;
TrainEVAL = compute_metric(pred_train, Y_train, [], option.metric_type);
TestEVAL  = compute_metric(pred_test, Y_test, [], option.metric_type);
fprintf('robustfit TrainTime=%.4f s | %s (%.4f %.4f) ||\n', ...
    t2, option.metric_type.name, TrainEVAL, TestEVAL);
end

model.n_hidden_nodes = Nh_nodes;
model.iter = iter;
model.InputWeight = iw;
model.BiasHidden = bias;
model.OutputWeight = out_w;
model.EVAL = [TrainEVAL TestEVAL];
model.Nh_nodes = Nh_nodes;
model.elm_type = elm_type;
model.rank_type = option.rank_type;
model.act_func = act_func;
model.metric_type = metric_type;
model.TrainTime = TrainingTime;
model.loss = loss;
model.c_rho = option.c_rho;

% -----------------------------
function s = madsigma(r,p)
%MADSIGMA    Compute sigma estimate using MAD of residuals from 0
rs = sort(abs(r));
s = median(rs(max(1,p):end)) / 0.6745;
