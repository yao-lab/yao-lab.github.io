function[W, Z, t_Z] = W_path_joint(X, D, y, nu, option)
% split_knockoffs.statistics.pathorder.W_fixed generate the knockoff
% statistics W for beta from a split LASSO path in the intercepetion
% assignment step, using the method of path order. 

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

% set lambda
lambda_vec = option.lambda_s;
nlambda = length(lambda_vec);

% set penalty
penalty = ones(m+p,1);
for i = 1: p
    penalty(i, 1) = 0;
end

% lasso path settings for glmnet
opts = struct; 
opts.lambda = lambda_vec;
opts.penalty_factor = penalty;
opts = glmnetSet(opts);

fit_step0 = glmnet([A_beta, A_gamma], tilde_y, [], opts);
coefs = fit_step0.beta;

% store beta(lambda)
betas = coefs(1: p, :);

%%%%%%%%%%%%% step 1 %%%%%%%%%%%%%%
coef1 = coefs(p+1: end, :);

% calculate r and Z
r = zeros(m, 1);
Z = zeros(m, 1);
for i = 1: m
    [Z(i), r(i)] = split_knockoffs.private.hittingpoint(coef1(i, :), lambda_vec);
end

W = Z;

%%%%%%%%%%%%% step 2 %%%%%%%%%%%%%% 
coef2 = zeros(2 * m, nlambda);
for i = 1: nlambda
    % take beta_lambda, gamma_lambda as calculated in step 1
    y_new = tilde_y - A_beta * betas(:, i);
    % calculate LASSO
    opts = struct; 
    opts.lambda = lambda_vec(i);
    opts = glmnetSet(opts);
    fit_step2 = glmnet([A_gamma, tilde_A_gamma], y_new, [], opts);
    coef2(:, i) = fit_step2.beta;
end  
% calculate tilde_Z tilde_r and W
t_Z = zeros(m, 1);
t_r = zeros(m, 1);

for i = 1: m
    [Z_dagger, ~] = split_knockoffs.private.hittingpoint(coef2(i, :), lambda_vec); 
    [tilde_Z, t_r(i)] = split_knockoffs.private.hittingpoint(coef2(i + m, :), lambda_vec); 

    if t_r(i) == r(i)
        % store tilde_Z when it is positive
        if tilde_Z > Z_dagger
            t_Z(i) = max(tilde_Z, Z(i));
            W(i) = - t_Z(i);
        elseif tilde_Z == Z_dagger
            t_Z(i) = Z(i);
            W(i) = 0;
        end
    end
end
end