function [H, InputWeight, BiasHidden] = elm_hidden_layer_gen(X, n_hidden_nodes, act_func, seed)
% Random generate input weights InputWeight (w_i) and biases BiasHidden (b_i) of hidden neurons

if nargin == 4
    rand('seed',seed);
end

n_input_neurons=size(X,2);
n_training_data=size(X,1);
InputWeight=rand(n_hidden_nodes,n_input_neurons)*2-1;
BiasHidden=rand(n_hidden_nodes,1);

H=InputWeight*X';

ind=ones(1,n_training_data);
BiasMatrix=BiasHidden(:,ind);              %   Extend the bias matrix BiasHidden to match the demention of H
H=H+BiasMatrix;

%% Calculate hidden neuron output matrix H
switch lower(act_func)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-H));
    case {'tanh'}
        %%%%%%%% tanh
        H = tanh(H);
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(H);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(H));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(H);
    case {'rbf','radbas'}
        %%%%%%%% Radial basis function
        H = radbas(H);
        %%%%%%%% More activation functions can be added here                
end
% clear tempH;

H = H';