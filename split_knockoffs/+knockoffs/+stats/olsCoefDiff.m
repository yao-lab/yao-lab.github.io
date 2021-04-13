function [W,Z] = olsCoefDiff(X, X_ko, y)
% KNOCKOFFS.STATS.OLSCOEFDIFF  The coefficient difference OLS statistic W
%   [W, Z] = KNOCKOFFS.STATS.OLSCOEFDIFF(X, X_ko, y)
%
%   Computes the statistic
%
%     W_j = |Z_j| - |\tilde Z_j|,
%
%   where Z_j and \tilde Z_j are the coefficient values of the 
%   jth variable and its knockoff, respectively, resulting from
%   OLS regression.
%
% See also KNOCKOFFS.STATS.LASSOCOEFDIFF, KNOCKOFFS.STATS.RIDGECOEFDIFF

p = size(X,2);

Z = glmfit([X X_ko],y,'normal');
Z = Z(2:end); % drop intercept

W = abs(Z(1:p))-abs(Z((p+1):(2*p)));

end