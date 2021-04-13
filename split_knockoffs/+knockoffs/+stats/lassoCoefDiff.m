function [W,Z] = lassoCoefDiff(X, X_ko, y, nfolds, cv)
% KNOCKOFFS.STATS.LASSOCOEFDIFF  Coefficient difference lasso statistic W 
% with cross-validation
%   [W, Z] = KNOCKOFFS.STATS.LASSOCOEFDIFF(X, X_ko, y)
%   [W, Z] = KNOCKOFFS.STATS.LASSOCOEFDIFF(X, X_ko, y, nfolds)
%   [W, Z] = KNOCKOFFS.STATS.LASSOCOEFDIFF(X, X_ko, y, nfolds, cv)
%
%   Computes the statistic
%
%     W_j = |Z_j| - |\tilde Z_j|,
%
%   where Z_j and \tilde Z_j are the coefficient values of the 
%   jth variable and its knockoff, respectively, resulting from
%   cross-validated lasso regression.
%
%   See also KNOCKOFFS.STATS.LASSOCOEFDIFF_BIN.

if ~exist('nfolds', 'var'), nfolds = []; end
if ~exist('cv', 'var'), cv = 'lambda_1se'; end

p = size(X,2);

options = glmnetSet();
options.standardize = true;
options.intr = true;
options.standardize_resp = false;
options.alpha = 1; % lasso regression

Z = cvglmnetCoef(cvglmnet([X X_ko],y,'gaussian',options,[],nfolds),cv);
Z = Z(2:end); % drop intercept

W = abs(Z(1:p))-abs(Z((p+1):(2*p)));

end