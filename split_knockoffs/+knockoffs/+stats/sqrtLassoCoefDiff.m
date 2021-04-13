function [W,Z] = sqrtLassoCoefDiff(X, X_ko, y, lambda)
% KNOCKOFFS.STATS.SQRTLASSOCOEFDIF  Signed difference of stability selection W
%   [W, Z] = SQRTLASSOCOEFDIF(X, X_ko, y)
%   [W, Z] = SQRTLASSOCOEFDIF(X, X_ko, y, lambda)
%
%   Computes the statistic
%
%     W_j = max(Z_j, \tilde Z_j) * sgn(Z_j - \tilde Z_j),
%
%   where Z_j and \tilde Z_j are the stability selection probabilities
%   values of the jth variable and its knockoff, respectively, resulting 
%   from fitting the SQRT-lasso.
%
% See also KNOCKOFFS.STATS.STABILITYSIGNEDMAX

m = 1000;
[n,p] = size(X);
alpha = 0.05;
kappa = 0.7;

if ~exist('lambda', 'var')
  eps = normrnd(0,1,n,m);
  S = [X X_ko]'*eps/n;
  Sinf = max(abs(S));
  lambda = kappa*n*quantile(Sinf,1-alpha);
end

[Z,~] = SqrtLassoIterative_WebPage([X X_ko], y, lambda, ones(2*p,1));

W = abs(Z(1:p))-abs(Z((p+1):(2*p)));

end