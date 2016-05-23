function model = robust_elm_train(X_train, Y_train, option)

%% input option
Nh_nodes           = option.Nh_nodes;
act_func = option.act_func;
metric_type        = option.metric_type;
c_rho              = 2^option.c_rho;
loss_type          = option.loss_type;
elm_type           = option.elm_type;
Max_iters          = option.Max_iters;

if isfield(option, 'tune')
    tune = option.tune;
end

if isfield(option, 'tune_relative')
    if strcmp(option.loss_type,'l1')
        default_tune = 1;
    elseif strcmp(option.loss_type,'huber')
        default_tune = 1.345;
        % option.tune = 0.9;
    elseif strcmp(option.loss_type,'bisquare')
        default_tune = 4.685;
    elseif strcmp(option.loss_type,'cauchy')
        default_tune = 2.385;
    elseif strcmp(option.loss_type,'welsch')
        default_tune = 2.985;
    end
    tune = option.tune_relative * default_tune;
end
    

if isfield(option, 'X_test')
    X_test = option.X_test;
    Y_test = option.Y_test;
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

[H, iw, bias] = elm_hidden_layer_gen(X_train, Nh_nodes, act_func);
if ~isempty(Y_test)
    H_test = elm_hidden_layer_apply(X_test, iw, bias, act_func);
end
tic;
% out_w = (speye(size(H,2))/c_rho + H'*H) \ (H'*T);
if strcmp(option.regu_type, 'none')
    if size(H,1) > size(H,2)
        out_w = pinv(H'*H) * (H'*T);
    else
        out_w = H'*pinv(H*H') * (T);
    end
else
    if size(H,1) > size(H,2)
        % out_w = (speye(size(H,2))/c_rho + H'*H) \ (H'*T);
        out_w = pinv(speye(size(H,2))/c_rho + H'*H) * (H'*T);
    else
        % out_w = H'*(speye(size(H,1))/c_rho + (H*H')) \ T;
        out_w = H'*pinv(speye(size(H,1))/c_rho + (H*H')) * T;
    end
end
Y = H*out_w;

if ~isempty(Y_test)
    pred_test = H_test * out_w;
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
lossfirst = TestEVAL;
% for iter=1:20
%     diff = H*out_w - T;
%     diff = diff/(max(tune,tiny_s)*tune);
%     W = robust_func(diff, 'l1', 'wgt');
%     W = sparse(W);
%     W = diag(W);
%     out_w = (speye(size(H,2))/c_rho + H'*W*H) \ (H'*(W*T));
% end
Y = H*out_w;
out_w_pre = out_w;
diff = Y - T;
s = madsigma(diff,1);
diff = diff/(max(s,tiny_s)*tune);
D = sqrt(eps(class(X_train)));
% loss = 0;
for iter=1:Max_iters
    if((iter~=1) && max(out_w-out_w_pre) <= 1e-6)
        % break
    end
    out_w_pre = out_w;
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
    W = sparse(W);
    W = diag(W);

    if size(H,1) >= size(H,2)
        if strcmp(option.regu_type, 'l1')
            W_beta = 1 ./ max(abs(out_w),0.000001);
            W_beta = diag(sparse(W_beta));
        elseif strcmp(option.regu_type, 'l2')
            W_beta = speye(size(H,2));
        end
        if strcmp(option.regu_type, 'none')
            out_w = pinv(H'*W*H) * (H'*(W*T));
        elseif strcmp(option.inv_type, 'inv')
            out_w = (W_beta/c_rho + H'*W*H) \ (H'*(W*T));
        elseif strcmp(option.inv_type, 'svd')
            out_w = pinv(W_beta/c_rho + H'*W*H) * (H'*(W*T));
        end
    else
        if strcmp(option.regu_type, 'l1')
            W_beta = abs(out_w);
            W_beta = diag(sparse(W_beta));
        elseif strcmp(option.regu_type, 'l2')
            W_beta = speye(size(H,2));
        end
        if strcmp(option.regu_type, 'none')
            out_w = H'*pinv(W*(H*H')) * (W*T);
        elseif strcmp(option.inv_type, 'inv')
            out_w = W_beta*H'*((speye(size(H,1))/c_rho + W*(H*W_beta*H')) \ (W*T));
        elseif strcmp(option.inv_type, 'svd')
            out_w = W_beta*H'*pinv(speye(size(H,1))/c_rho + W*(H*W_beta*H')) * (W*T);
        end
    end
    
    Y = H*out_w;

    %% 
    pred_train = Y;
    if ~isempty(Y_test)
        pred_test = H_test * out_w;
    end
    valid_interval = 1;
    if strcmp(option.regu_type, 'l1')
        train_loss = 1/c_rho*0.5*sum(abs(out_w))+s*s*sum(robust_func(diff, loss_type, 'rho', 1));
    elseif strcmp(option.regu_type, 'l2')
        train_loss = 1/c_rho*0.5*sum(out_w.^2)+s*s*sum(robust_func(diff, loss_type, 'rho', 1));
    end
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
            ki = iter;
%             ki = (k - 1)*Max_iters+iter/valid_interval;
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

if 0 & size(H,2)<=200
t1=tic;
brob = robustfit(H, Y_train, option.loss_type);
t2=toc(t1);
pred_train = [ones(size(H,1),1) H] * brob;
pred_test =  [ones(size(H_test,1),1) H_test] * brob;
TrainEVAL = compute_metric(pred_train, Y_train, [], option.metric_type);
if ~isempty(Y_test)
    TestEVAL  = compute_metric(pred_test, Y_test, [], option.metric_type);
end
fprintf('robustfit TrainTime=%.4f s | %s (%.4f %.4f) ||\n', ...
    t2, option.metric_type.name, TrainEVAL, TestEVAL);
end

model.n_hidden_nodes = Nh_nodes;
model.N = Nh_nodes;
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
model.lossfirst = lossfirst;
model.c_rho = option.c_rho;

% -----------------------------
function s = madsigma(r,p)
%MADSIGMA    Compute sigma estimate using MAD of residuals from 0
rs = sort(abs(r));
s = median(rs(max(1,p):end)) / 0.6745;
