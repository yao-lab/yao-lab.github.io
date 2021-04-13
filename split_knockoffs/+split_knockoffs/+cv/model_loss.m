function CV_loss = model_loss(X, y, model_s, option)
% split_knockoffs.cv.model_loss calculate the cv loss for linear regression
% w.r.t. different sparse model.

% input arguments
% X : the design matrix
% y : the response vector
% model_s: the sparse models to calculate the loss
% option: options for creating the Knockoff statistics
% % option.k_fold: the fold used in cross validation

% output arguments
% CV_loss: CV loss w.r.t. each model
    
num_model = length(model_s);
CV_loss = zeros(num_model, 1);
k_fold = option.k_fold;

[n, p] = size(X);
test_size = floor(n / k_fold);
% generate random order of sample
rng(1);
rand_rank = randperm(n);

for k = 1: k_fold
    % determine the index for test set
    test_index = [floor(test_size * (k-1) + 1): floor(test_size * k)];
    test = rand_rank(test_index);

    % calculate training set and test set
    X_train = X;
    X_train(test, :) = [];
    y_train = y;
    y_train(test, :) = [];

    X_test = X(test, :);
    y_test = y(test, :);
   
    % normalization
    X_train = bsxfun(@minus,X_train,mean(X_train,1));
    y_train = bsxfun(@minus,y_train,mean(y_train,1));
    X_test = bsxfun(@minus,X_test,mean(X_test,1));
    y_test = bsxfun(@minus,y_test,mean(y_test,1));
    
    loss = zeros(num_model, 1);
    
    for i = 1: num_model
        % find the estimated non-zero set index
        index = model_s{i};
        null_set = eye(p);
        null_set(index, :) = [];
        b = zeros(p-length(index), 1);
        opts=  optimset('display','off');
        beta_optimal = lsqlin(X_train, y_train,[],[], null_set, b, [], [],[], opts);
        
        % make prediction and store the CV loss
        y_predict = X_test * beta_optimal;
        loss(i) = (norm(y_predict - y_test))^2 / length(y_test);
    end
    CV_loss = CV_loss + loss / k_fold;
end
end