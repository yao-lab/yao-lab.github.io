function first_lambda = lassoMaxLambda_probit(X, y, nlambda)
% KNOCKOFFS.STATS.PRIVATE.LASSOMAXLAMBDA_PROBIT  Maximum lambda's for which 
% variables in lasso model (probit response)
%   maxLambda = KNOCKOFFS.STATS.PRIVATE.LASSOMAXLAMBDA_PROBIT(X, y)
%   maxLambda = KNOCKOFFS.STATS.PRIVATE.LASSOMAXLAMBDA_PROBIT(X, y, nlambda)
%
%   For each variable (column in X), computes the maximum value of lambda 
%   at which the variable enters in the lasso logistic regression model.
%   %%%CANNOT REMOVE INTERCEPT USING LASSOGLM%%%

[n,p] = size(X);
if ~exist('nlambda', 'var') || isempty(nlambda)
    nlambda = 200;
end
lambdaratio = 1/(2e3);

X = knockoffs.private.normc(X);  % Standardize the variables

[B,FitInfo] = lassoglm(X,y,'binomial','Link','probit','Standardize',false,'LambdaRatio',lambdaratio,'NumLambda',nlambda);
if size(B,2)<nlambda, B = [repmat(B(:,1),1,nlambda-size(B,2)) B]; end
B = B(:,nlambda:-1:1);
lambdas = FitInfo.Lambda(end)*lambdaratio.^((0:(nlambda-1))/(nlambda-1));
first_lambda = zeros(1,p);
for j = 1:p
    first_time = find(abs(B(j,:)) > 0, 1, 'first');
    if isempty(first_time)
        first_lambda(j) = 0;
    else
        first_lambda(j) = lambdas(first_time);
    end
end

% lassoglm uses non-standard scaling of lambda.
first_lambda = first_lambda * n;

end