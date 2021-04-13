function [results, Z, t_Z] = filter(X, D, y, nu_s, option)
% split Knockoff filter for structural sparsity problem.

% input argument
% X : the design matrix
% y : the response vector
% D : the linear transform
% nu_s: a set of nu, appointed by the user
% nu: the parameter for variable splitting
% option: options for creating the Knockoff statistics
% % option.eta : the choice of eta for creating the knockoff copy
% % option.q: the desired FDR control bound
% % option.method: "knockoff" or "knockoff+"
% % option.stage0: choose the method to conduct split knockoff
% %     "fixed": fixed intercept assignment for PATH ORDER method
% %         option.beta : the choice of fixed beta for step 0: 
% %             "mle": maximum likelihood estimator; 
% %             "ridge": ridge regression choice beta with lambda = 1/nu
% %             "cv_split": cross validation choice of split LASSO over nu and lambda
% %             "cv_ridge": cross validation choice of ridge regression over lambda
% %     "path": take the regularization path of split LASSO as the intercept
% %             assignment for PATH ORDER method
% %     "magnitude": using MAGNITUDE method
% % option.lambda_s: the choice of lambda for path
% % option.normalize: whether to normalize the data

% output argument
% results: a cell of length(nu_s) * 1 with the selected variable set in each
% cell w.r.t. nu.
% Z: a cell of length(nu_s) * 1 with the feature significance Z in each
% cell w.r.t. nu.
% t_Z: a cell of length(nu_s) * 1 with the knockoff significance tilde_Z in each
% cell w.r.t. nu.

if option.normalize == true
    X = normalize(X);
    y = normalize(y);
end


num_nu = length(nu_s);
results = cell(num_nu, 1);
Z = cell(num_nu, 1);
t_Z = cell(num_nu, 1);

[n, ~] = size(X);

q = option.q;
method = option.method;
stage0 = option.stage0;

n_nu = length(nu_s);

if stage0 == "fixed" && option.beta ~= "ridge"
    option.beta_choice = split_knockoffs.statistics.pathorder.fixed_beta(X, y, D, option);
end

for i = 1: n_nu
    nu = nu_s(i);
    switch stage0
        case "fixed"
            if option.beta == "ridge"
                option.beta_choice = (X' * X / n  + 1 / nu * D' * D)^-1 * X' * y / n;
            end
            [W, Z{i}, t_Z{i}] = split_knockoffs.statistics.pathorder.W_fixed(X, D, y, nu, option);
        case "path"
            [W, Z{i}, t_Z{i}] = split_knockoffs.statistics.pathorder.W_path(X, D, y, nu, option);
        case "magnitude"
            [W, Z{i}, t_Z{i}] = split_knockoffs.statistics.magnitude.W_mag(X, D, y, nu, option);
        case "joint"
            [W, Z{i}, t_Z{i}] = split_knockoffs.statistics.pathorder.W_path_joint(X, D, y, nu, option);
    end
    results{i} = knockoffs.select(W, q, method);
end
end
