function [W,Z] = ridgeCoefDiff(X, X_ko, y, nfolds)
% KNOCKOFFS.STATS.RIDGECOEFDIFF  Ccoefficient difference ridge statistic W
%   [W, Z] = KNOCKOFFS.STATS.RIDGECOEFDIFF(X, X_ko, y)
%   [W, Z] = KNOCKOFFS.STATS.RIDGECOEFDIFF(X, X_ko, y, nfolds)
%
%   Computes the statistic
%
%     W_j = Z_j - \tilde Z_j,
%
%   where Z_j and \tilde Z_j are the coefficient values of the 
%   jth variable and its knockoff, respectively, resulting from
%   cross-validated ridge regression.
%
% See also KNOCKOFFS.STATS.LASSOCOEFDIFF, KNOCKOFFS.STATS.OLSCOEFDIFF, KNOCKOFFS.STATS.RIDGECOEFDIFF_BIN

if ~exist('nfolds', 'var'), nfolds = []; end

p = size(X,2);

options = glmnetSet();
options.standardize = true;
options.intr = true;
options.standardize_resp = false;
options.alpha = 0; % ridge regression

Z = cvglmnetCoef(cvglmnet([X X_ko],y,'gaussian',options,[],nfolds)); %uses default 1se rule
Z = Z(2:end); % drop intercept

W = abs(Z(1:p))-abs(Z((p+1):(2*p)));

end