function [W, Z] = lassoLambdaDifference_bin(X, X_ko, y, nlambda)
% KNOCKOFFS.STATS.LASSOLAMBDADIFFERENCEBIN  Difference lasso statistic W
%   [W, Z] = KNOCKOFFS.STATS.LASSOLAMBDADIFFERENCEBIN(X, X_ko, y)
%   [W, Z] = KNOCKOFFS.STATS.LASSOLAMBDADIFFERENCEBIN(X, X_ko, y, nlambda)
%
%   Computes the statistic
%
%     W_j = Z_j - \tilde Z_j,
%
%   where Z_j and \tilde Z_j are the maximum values of the regularization
%   parameter lambda at which the jth variable and its knockoff,
%   respectively, enter the logistic regression model with L1 penalty.
%
%   See also KNOCKOFFS.STATS.LASSOLAMBDADIFFERENCE.

if ~exist('nlambda', 'var'), nlambda = []; end

Z = knockoffs.stats.private.lassoMaxLambda_binom([X X_ko], y, nlambda);

p = size(X,2);
orig = 1:p; ko = (p+1):(2*p);
W = Z(orig) - Z(ko);

end