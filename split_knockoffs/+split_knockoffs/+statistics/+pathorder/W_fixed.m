function[W, Z, t_Z] = W_fixed(X, D, y, nu, option)
% split_knockoffs.statistics.pathorder.W_fixed generate the knockoff
% statistics W for fixed beta in the intercepetion assignment step, using
% the method of path order.

% input argument:
% X : the design matrix
% y : the response vector
% D : the linear transform
% nu: the parameter for variable splitting
% option: options for creating the Knockoff statistics
% % option.eta : the choice of eta for creating the knockoff copy
% % option.lambda_s: the choice of lambda for the path
% % option.beta_choice : the fixed beta for step 0: 

% output argument
% W: the knockoff statistics
% Z: feature significance
% t_Z: knockoff significance

m = size(D, 1);

opts.copy = true;
opts.eta = option.eta;

% generate the design matrix
[A_beta,A_gamma,tilde_y,tilde_A_gamma] = split_knockoffs.create(X, y, D, nu, opts);

%%%%%%%%%%%%% step 0 %%%%%%%%%%%%%%

beta_hat = option.beta_choice;

% calculate new response vector
y_new = tilde_y - A_beta * beta_hat;

% appoint a set of lambda
lambda_vec = option.lambda_s;

%%%%%%%%%%%%% step 1 %%%%%%%%%%%%%%
opts = struct; 
if isempty(lambda_vec) == false
    opts.lambda = lambda_vec;
end
opts = glmnetSet(opts);

fit_step1 = glmnet(A_gamma, y_new, [], opts);
lambda_vec = fit_step1.lambda;
coef1 = fit_step1.beta;

% calculate r and Z
r = zeros(m, 1);
Z = zeros(m, 1);
for i = 1: m
    [Z(i), r(i)] = split_knockoffs.private.hittingpoint(coef1(i, :), lambda_vec);
end

%%%%%%%%%%%%% step 2 %%%%%%%%%%%%%% 
% lasso path settings for glmnet
opts = struct; 
opts.lambda = lambda_vec;
opts = glmnetSet(opts);

fit_step2 = glmnet(tilde_A_gamma, y_new, [], opts);
coef2 = fit_step2.beta;

% calculate tilde_Z tilde_r
t_Z = zeros(m, 1);
t_r = zeros(m, 1);

for i = 1: m
    [tilde_Z, t_r(i)] = split_knockoffs.private.hittingpoint(coef2(i, :), lambda_vec); 
    if t_r(i) == r(i)
        % store tilde_Z when it has the same sign
        t_Z(i) = tilde_Z;
    end
end

%%%%%%%%%%%%% W %%%%%%%%%%%%%% 
W = max(Z, t_Z) .* sign(Z - t_Z);
end