function lambda = cv_mag(X, y, D, nu, option)
% split_knockoffs.statistics.magnitude.cv_mag calculate the CV optimal
% lambda in the problem 
% % 1/n |y - X beta|^2 + 1/nu |D beta - gamma|^2 + lambda |gamma|_1
% for a fixed nu.

% input argument
% X : the design matrix
% y : the response vector
% D : the linear transform
% nu: the parameter for variable splitting
% option: options for calculating cv
% % option.lambda_s: the choice of lambda for the path

% output argument
% lambda: CV optimal lambda


k_fold = 10;
[n, ~] = size(X);
[m, p] = size(D);

% appoint a set of lambda
if isempty(option.lambda_s) == false
    lambda_s = option.lambda_s;
    nlambda = length(lambda_s);
else
    power = [0:-0.1:-10];
    lambda_s = 10.^power;
    nlambda = length(lambda_s);
end

test_size = floor(n / k_fold);

% randomly split data
rng(1);
rand_rank = randperm(n);

% create matrix to store result for split lasso test loss
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
    
    % try normalization
    X_train = bsxfun(@minus,X_train,mean(X_train,1));
    y_train = bsxfun(@minus,y_train,mean(y_train,1));
    X_test = bsxfun(@minus,X_test,mean(X_test,1));
    y_test = bsxfun(@minus,y_test,mean(y_test,1));
        
    % generate split LASSO design matrix
    [n_train, ~] = size(X_train);
    A_beta = [X_train/sqrt(n_train);D/sqrt(nu)];
    A_gamma = [zeros(n_train,m);-eye(m)/sqrt(nu)];
    tilde_y = [y_train/sqrt(n_train);zeros(m,1)];

    % set penalty for glmnet
    penalty = ones(m+p,1);
    for tmp = 1: p
        penalty(tmp) = 0;
    end

    opts.standardize = false; % cancel standardize for glmnet
    opts.lambda = lambda_s; % choose lambda
    opts.intr = false;
    opts.penalty_factor = penalty; % choose penalty
    options = glmnetSet(opts);

    % fit glmnet
    fit = glmnet([A_beta, A_gamma], tilde_y, [], options);
    % extract beta
    coefs = fit.beta;

    for j =  1: length(lambda_s)
        coef = coefs(:, j);
        beta = coef(1:p);
        % calculate loss
        y_sl = X_test * beta;
        loss_sl(k, j) = norm(y_sl-y_test)^2/test_size;
    end
    
end

mean_loss_sl = mean(loss_sl, 1);

% find minimal
lambda_number = find(mean_loss_sl == min(min(mean_loss_sl)), 1);
lambda = lambda_s(lambda_number);
end