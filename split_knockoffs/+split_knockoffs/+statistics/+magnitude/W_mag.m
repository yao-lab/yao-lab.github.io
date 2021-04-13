function[W, Z, t_Z] = W_mag(X, D, y, nu, option)
% split_knockoffs.statistics.pathorder.W_mag generate the knockoff
% statistics W, using the method of magnitude. lambda here is chosen by
% cross validation.

% input argument:
% X : the design matrix
% y : the response vector
% D : the linear transform
% nu: the parameter for variable splitting
% option: options for creating the Knockoff statistics
% % option.eta : the choice of eta for creating the knockoff copy
% % option.lambda_s: the choice of lambda for the path

% output argument
% W: the knockoff statistics
% Z: feature significance
% t_Z: knockoff significance

[m, p] = size(D);

opts.copy = true;
opts.eta = option.eta;

% generate the design matrix
[A_beta,A_gamma,tilde_y,tilde_A_gamma] = split_knockoffs.create(X, y, D, nu, opts);

%%%%%%%%%%%%% step 0 %%%%%%%%%%%%%%

lambda = split_knockoffs.statistics.magnitude.cv_mag(X, y, D, nu, option);
opts = struct; 
opts.lambda = lambda;
opts = glmnetSet(opts);

fit_step0 = glmnet([A_beta, A_gamma], tilde_y, [], opts);
coef = fit_step0.beta;
beta_hat = coef(1:p);
y_new = tilde_y - A_beta * beta_hat;


%%%%%%%%%%%%% step 1 %%%%%%%%%%%%%%
opts = struct; 
opts.lambda = lambda;
opts = glmnetSet(opts);

fit_step1 = glmnet(A_gamma, y_new, [], opts);
coef1 = fit_step1.beta;

% calculate r and Z
Z = abs(coef1);
r = sign(coef1);

%%%%%%%%%%%%% step 2 %%%%%%%%%%%%%% 
% lasso path settings for glmnet
opts = struct; 
opts.lambda = lambda;
opts = glmnetSet(opts);

fit_step2 = glmnet(tilde_A_gamma, y_new, [], opts);
coef2 = fit_step2.beta;

% calculate tilde_Z tilde_r
t_Z = zeros(m, 1);
Z_prime = abs(coef2);
t_r = sign(coef2);

for i = 1: m
    if t_r(i) == r(i)
        % store tilde_Z when it has the same sign
        t_Z(i) = Z_prime(i);
    end
end

%%%%%%%%%%%%% W %%%%%%%%%%%%%% 
W = max(Z, t_Z) .* sign(Z - t_Z);
end