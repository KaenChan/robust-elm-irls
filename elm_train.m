function model = elm_train(X_train, Y_train, option)

%% input option
n_hidden_nodes = option.n_hidden_nodes;
act_func       = option.act_func;
c_rho          = 2^option.c_rho;
elm_type       = option.elm_type;

% elm_type = 'regression';
% elm_type = 'classifier';

if isfield(option, 'seed')
    seed = option.seed;
else
    seed = fix(mod(cputime,100));
end
rand('seed',seed);

%% Woad training dataset

T=Y_train;

NumberofTrainingData=size(X_train,1);
NumberofInputNeurons=size(X_train,2);

T=double(T);

t1=tic;

%% Calculate weights & biases

[H, InputWeight, BiasHidden] = elm_hidden_layer_gen(X_train, n_hidden_nodes, act_func, seed);

%% Calculate output weights OutputWeight (beta_i)
n = n_hidden_nodes;

% OutputWeight=((H'*H+(eye(n)/c_rho))\(H'*T)); 

if size(H,1) > size(H,2)
    HH = H'*H;
    HT = H'*T;
    OutputWeight=((HH+(eye(n)/c_rho))\(HT)); 
else
    HH = H*H';
    OutputWeight=H'*((HH+(eye(size(H,1))/c_rho))\(T)); 
end

TrainTime = toc(t1);

% TrainTime=toc;
%%%%%%%%%%% Calculate the training accuracy
pred = (H * OutputWeight);

if strcmp(elm_type, 'classifier')
    %%%%%%%%%% Calculate training & testing classification accuracy
    missclassified=0;

    for i = 1 : size(X_train, 1)
        [x, label_index_expected]=max(pred(i,:));
        [x, label_index_actual]=max(Y_train(i,:));
        if label_index_actual ~= label_index_expected
            missclassified = missclassified + 1;
        end
    end
    TrainEVAL = 1-missclassified/NumberofTrainingData;
elseif strcmp(elm_type, 'regression')
    TrainEVAL = sqrt(mse(Y_train - pred));
end

model.elm_type = elm_type;
model.InputWeight = InputWeight;
model.n_hidden_nodes = n_hidden_nodes;
model.N = n_hidden_nodes;
model.BiasHidden = BiasHidden;
model.OutputWeight = OutputWeight;
model.act_func = act_func;
model.c_rho = option.c_rho;
model.TrainTime = TrainTime;
model.TrainEVAL = TrainEVAL;
