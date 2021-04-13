function [W, Z] = forwardSelectionOMP(X, X_ko, y)
% KNOCKOFFS.STATS.FORWARDSELECTIONOMP  Forward selection statistic with OMP
%   [W, Z] = KNOCKOFFS.STATS.FORWARDSELECTIONOMP(X, X_ko, y)
%
%   This variant of forward selection uses orthogonal matching pursuit
%   (OMP).
%
%   See also KNOCKOFFS.STATS.FORWARDSELECTION.

added = knockoffs.stats.private.forwardSelectionOMP([X X_ko], y);
[~,order] = sort(added);

p = size(X,2);
Z = 2*p + 1 - order;
orig = 1:p; ko = (p+1):(2*p);
W = max(Z(orig), Z(ko)) .* sign(Z(orig) - Z(ko));

end