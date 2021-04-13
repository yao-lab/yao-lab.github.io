function beta = cv_ridge(X, y, D)
% split_knockoffs.statistics.pathorder.cv_ridge calculate the CV optimal beta
% in the problem 1/n |y - X beta|^2 + lambda |D beta|^2.

% input argument
% X : the design matrix
% y : the response vector
% D : the linear transform

% output argument
% beta: CV optimal beta

k_fold = 10;
[n, p] = size(X);

% appoint a set of lambda
power = [1:-0.1:-10];
lambda_s = 10.^power;
nlambda = length(lambda_s);

test_size = floor(n / k_fold);
% randomly split data
rng(1);
rand_rank = randperm(n);

% create matrix to store result for test loss
loss_sl = zeros(k_fold, nlambda);


for k = 1: k_fold
    
    % generate test set
    test_index = [test_size * (k-1) + 1: test_size * k];
    test = rand_rank(test_index);
    
    % training set
    X_train = X;
    X_train(test, :) = [];
    y_train = y;
    y_train(test, :) = [];
    
    % test set
    X_test = X(test, :);
    y_test = y(test, :);
    
    % normalization
    X_train = bsxfun(@minus,X_train,mean(X_train,1));
    y_train = bsxfun(@minus,y_train,mean(y_train,1));
    X_test = bsxfun(@minus,X_test,mean(X_test,1));
    y_test = bsxfun(@minus,y_test,mean(y_test,1));
    
    betas = zeros(p, nlambda);
    
    % solve ridge regression
    for i = 1: nlambda
        lambda = lambda_s(i);
        betas(:, i) = (X_train' * X_train / n + lambda * D' * D)^-1 * X_train' * y_train / n;
    end
    

    for j =  1: nlambda
        beta = betas(:, j);
        % calculate loss
        y_sl = X_test * beta;
        loss_sl(k, j) = norm(y_sl-y_test)^2/test_size;
    end
    
end

mean_loss_sl = mean(loss_sl, 1);

% find minimal
[lambda_number] = find(mean_loss_sl == min(mean_loss_sl), 1);
lambda_sl = lambda_s(lambda_number);

% calculate beta

beta = (X' * X / n  + lambda_sl * D' * D)^-1 * X' * y / n;
end