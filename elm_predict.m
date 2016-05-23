function [pred, TestTime, TestEVAL] = elm_predict(model, Xt, Yt)

%     load rank_elm_model.mat;
    InputWeight = model.InputWeight;
    BiasHidden = model.BiasHidden;
    OutputWeight = model.OutputWeight;
    act_func = model.act_func;

    %%%%%%%%%%% Calculate the output of testing input
    t1=clock;

    H_test = elm_hidden_layer_apply(Xt, InputWeight, BiasHidden, act_func);

    pred=(H_test * OutputWeight);
    
    TestEVAL = 0;
    if ~isempty(Yt)
        TestEVAL = evaluation_preds(pred, Yt, model.elm_type);
    end

    t2 = clock;
    TestTime = etime(t2,t1);
