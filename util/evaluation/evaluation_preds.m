function eva = evaluation_preds(pred, Y, elm_type)
    if strcmp(elm_type, 'classifier')
        %%%%%%%%%% Calculate training & testing classification accuracy
        missclassified=0;
        if size(Y,2)>1
            for i = 1 : size(Y, 1)
                [x, label_index_expected]=max(pred(i,:));
                [x, label_index_actual]=max(Y(i,:));
                if label_index_actual ~= label_index_expected
                    missclassified = missclassified + 1;
                end
            end
            eva = 1-missclassified/size(Y,1);
        else
            eva = mean(pred==Y);
        end
    elseif strcmp(elm_type, 'binary')
        label_expected = ones(size(pred));
        label_expected(pred<0) = -1;
        eva = mean(label_expected==Y);
    elseif strcmp(elm_type, 'regression')
        eva = sqrt(mse(Y - pred));
    end
