function [W,Z] = ridgeCoefDiff_bin(X, X_ko, y, nfolds)
% KNOCKOFFS.STATS.RIDGECOEFDIFF_BIN  Coefficient difference ridge statistic 
% W (binomial response)
%   [W, Z] = KNOCKOFFS.STATS.RIDGECOEFDIFF_BIN(X, X_ko, y)
%   [W, Z] = KNOCKOFFS.STATS.RIDGECOEFDIFF_BIN(X, X_ko, y, nfolds)
%
%   Computes the statistic
%
%     W_j = Z_j - \tilde Z_j,
%
%   where Z_j and \tilde Z_j are the coefficient values of the 
%   jth variable and its knockoff, respectively, resulting from
%   cross-validated logistic regression with L2 regularization.
%
% See also KNOCKOFFS.STATS.RIDGECOEFDIFF

if ~exist('nfolds', 'var'), nfolds = []; end

p = size(X,2);

options = glmnetSet();
options.standardize = true;
options.intr = true;
options.standardize_resp = false;
options.alpha = 0; % ridge regression
lambda_max = max(abs(X'*(y-1/2)))/size(X,1);
lambda_min = lambda_max/(2*1e3);
nlambda = 100;
k = (0:(nlambda-1))/nlambda;
options.lambda = lambda_max .* (lambda_min/lambda_max).^k;

Z = cvglmnetCoef(cvglmnet([X X_ko],y,'binomial',options,[],nfolds)); %uses default 1se rule
Z = Z(2:end); % drop intercept

W = abs(Z(1:p))-abs(Z((p+1):(2*p)));

end