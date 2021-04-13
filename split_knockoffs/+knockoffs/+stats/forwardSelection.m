function [W, Z] = forwardSelection(X, X_ko, y)
% KNOCKOFFS.STATS.FORWARDSELECTION  Forward selection statistic W
%   [W, Z] = KNOCKOFFS.STATS.FORWARDSELECTION(X, X_ko, y)
%
%   Computes the statistic
%
%     W_j = max(Z_j, Z_{j+p}) * sgn(Z_j - Z_{j+p})
%
%   where Z_1,\dots,Z_{2p} give the reverse order in which the 2p variables
%   (the originals and the knockoffs) enter the forward selection model.
%
%   See also KNOCKOFFS.STATS.FORWARDSELECTIONOMP.

added = knockoffs.stats.private.forwardSelection([X X_ko], y);
[~,order] = sort(added);

p = size(X,2);
Z = 2*p + 1 - order;
orig = 1:p; ko = (p+1):(2*p);
W = max(Z(orig), Z(ko)) .* sign(Z(orig) - Z(ko));

end