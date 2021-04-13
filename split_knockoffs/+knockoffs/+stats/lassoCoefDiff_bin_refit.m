function [W,Z] = lassoCoefDiff_bin_refit(X, X_ko, y, nfolds)
% KNOCKOFFS.STATS.LASSOCOEFDIFF_BIN_REFIT Coefficient difference lasso 
% statistic W with cross-validated lambda (binomial response)
%   [W, Z] = KNOCKOFFS.STATS.LASSOCOEFDIFF_BIN_REFIT(X, X_ko, y)
%   [W, Z] = KNOCKOFFS.STATS.LASSOCOEFDIFF_BIN_REFIT(X, X_ko, y, nfolds)
%
%   Computes the statistic
%
%     W_j = |Z_j| - |\tilde Z_j|,
%
%   where Z_j and \tilde Z_j are the coefficient values of the 
%   jth variable and its knockoff, respectively, resulting from
%   cross-validated logistic regression with L1 regularization.
%   The coefficients are obtained from a re-fitted model (on the full data) 
%   after lambda has been selected by cross-validation.
%
%   See also KNOCKOFFS.STATS.LASSOCOEFDIFF, KNOCKOFFS.STATS.LASSOCOEFDIFF_BIN.

if ~exist('nfolds', 'var'), nfolds = []; end

p = size(X,2);

options = glmnetSet();
options.standardize = true;
options.intr = true;
options.standardize_resp = false;
options.alpha = 1; % lasso regression

fit = cvglmnet([X X_ko],y,'binomial',options,[],nfolds);
lambda = max(fit.lambda(fit.cvm<=min(fit.cvup)));
Z = cvglmnetCoef(fit,lambda);
Z = Z(2:end); % drop intercept

W = abs(Z(1:p))-abs(Z((p+1):(2*p)));

end