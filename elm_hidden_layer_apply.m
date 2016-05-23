function H = elm_hidden_layer_apply(X, InputWeight, BiasHidden, act_func)
% Random generate input weights InputWeight (w_i) and biases BiasHidden (b_i) of hidden neurons

NumberofTestingData=size(X,1);

H=InputWeight*X';
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasHidden(:,ind);              %   Extend the bias matrix BiasHidden to match the demention of H
H = H + BiasMatrix;
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
        H = hardlim(H);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(H);        
    case {'rbf','radbas'}
        %%%%%%%% Radial basis function
        H = radbas(H);        
        %%%%%%%% More activation functions can be added here        
end
H = H';
