function first_lambda = lassoMaxLambda_binom(X, y, nlambda)
% KNOCKOFFS.STATS.PRIVATE.LASSOMAXLAMBDA_BINOM  Maximum lambda's for which 
% variables in lasso model (binomial response)
%   maxLambda = KNOCKOFFS.STATS.PRIVATE LASSOMAXLAMBDA_BINOM(X, y)
%   maxLambda = KNOCKOFFS.STATS.PRIVATE.LASSOMAXLAMBDA_BINOM(X, y, nlambda)
%
%   For each variable (column in X), computes the maximum value of lambda 
%   at which the variable enters in the lasso logistic regression model.

[n,p] = size(X);
if ~exist('nlambda', 'var') || isempty(nlambda)
    nlambda = 200;
end

X = knockoffs.private.normc(X);  % Standardize the variables

options = glmnetSet();
options.standardize = false;
options.intr = true;
options.standardize_resp = false;

lambda_max = max(abs(X'*(0.5-y)))/n;
lambda_min = lambda_max/(2*1e3);
k = (0:(nlambda-1))/nlambda;
options.lambda = lambda_max .* (lambda_min/lambda_max).^k;

fit = glmnet(X,y,'binomial',options); %diff 2
first_lambda = zeros(1,p);
for j = 1:p
    first_time = find(abs(fit.beta(j,:)) > 0, 1, 'first');
    if isempty(first_time)
        first_lambda(j) = 0;
    else
        first_lambda(j) = fit.lambda(first_time);
    end
end

% glmnet uses non-standard scaling of lambda.
first_lambda = first_lambda * n;

end