function[results, CV_loss, complexities] = loss(X, D, y, nu_s, option)
% split_knockoffs.cv.loss calculate the cv loss for split Knockoffs
% w.r.t. different nu.

% input arguments
% X : the design matrix
% y : the response vector
% D : the linear transform
% nu_s: a set of nu, appointed by the user
% option: options for creating the Knockoff statistics
% % option.k_fold: the fold used in cross validation

% output arguments
% results: selection sets for all nu in nu_s
% CV_loss: CV loss w.r.t. each choice of nu
% complexities: model complexities for split Knockoff w.r.t. different nu
    
num_nu = length(nu_s);
CV_loss = zeros(num_nu, 1);
complexities = zeros(num_nu, 1);
k_fold = option.k_fold;

results = split_knockoffs.filter(X, D, y, nu_s, option);

% runing original and alternative step 2 knockoff
for i = 1: num_nu
    complexities(i) = length(results{i});
end

[n, ~] = size(X);
[m, ~] = size(D);
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
    
    option_cv = option;
    option_cv.normalization = false;
    results_cv = split_knockoffs.filter(X_train, D, y_train, nu_s, option_cv);
    
    loss = zeros(num_nu, 1);
    
    for i = 1: num_nu
        % find the estimated non-zero set index
        index = results_cv{i};
        D_null = D;
        D_null(index, :) = [];
        b = zeros(m-length(index), 1);
        opts=  optimset('display','off');
        beta_optimal = lsqlin(X_train, y_train,[],[], D_null, b, [], [],[], opts);
        
        % make prediction and store the CV loss
        y_predict = X_test * beta_optimal;
        loss(i) = (norm(y_predict - y_test))^2 / length(y_test);
    end
    CV_loss = CV_loss + loss / k_fold;
end
end