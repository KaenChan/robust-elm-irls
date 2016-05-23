function acc = compute_acc(pred,Y)
missclassified=0;
if size(Y,2)>1
    for i = 1 : size(Y, 1)
        [x, label_index_expected]=max(pred(i,:));
        [x, label_index_actual]=max(Y(i,:));
        if label_index_actual ~= label_index_expected
            missclassified = missclassified + 1;
        end
    end
    acc = 1-missclassified/size(Y,1);
else
    % label_expected = ones(size(pred));
    % label_expected(pred<0) = -1;
    % acc = mean(label_expected==Y);
    acc = mean(pred==Y);
end