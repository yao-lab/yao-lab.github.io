function [W,Z] = lassoCoefDiff_bin(X, X_ko, y, nfolds)
% KNOCKOFFS.STATS.LASSOCOEFDIFF_BIN  Coefficient difference lasso statistic 
% W with cross-validation (binomial response)
%   [W, Z] = KNOCKOFFS.STATS.LASSOCOEFDIFF_BIN(X, X_ko, y)
%   [W, Z] = KNOCKOFFS.STATS.LASSOCOEFDIFF_BIN(X, X_ko, y, nfolds)
%
%   Computes the statistic
%
%     W_j = |Z_j| - |\tilde Z_j|,
%
%   where Z_j and \tilde Z_j are the coefficient values of the 
%   jth variable and its knockoff, respectively, resulting from
%   cross-validated logistic regression with L1 regularization.
%
%   See also KNOCKOFFS.STATS.LASSOCOEFDIFF, KNOCKOFFS.STATS.LASSOCOEFDIFF_BIN_REFIT.

if ~exist('nfolds', 'var'), nfolds = []; end

p = size(X,2);

options = glmnetSet();
options.standardize = true;
options.intr = true;
options.standardize_resp = false;
options.alpha = 1; % lasso regression

Z = cvglmnetCoef(cvglmnet([X X_ko],y,'binomial',options,[],nfolds)); %uses default 1se rule
Z = Z(2:end); % drop intercept

W = abs(Z(1:p))-abs(Z((p+1):(2*p)));

end