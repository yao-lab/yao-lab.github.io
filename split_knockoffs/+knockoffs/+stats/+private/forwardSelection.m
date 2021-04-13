function varargout = forwardSelection(X, y)
% KNOCKOFFS.STATS.PRIVATE.FORWARDSELECTION  Fast implementation of forward selection
%
%   Assumes that the columns of X are normalized to 1.
%
% See also KNOCKOFFS.STATS.PRIVATE.FORWARDSELECTIONOMP

X = knockoffs.private.normc(X);  % Standardize the variables

[varargout{1:nargout}] = ...
    knockoffs.stats.private.sequentialfs(@criterion, @target, X, y);

end

function c = criterion(~, x, residual)
    c = -abs(dot(x, residual));
end

function nextResidual = target(~, x, residual)
    nextResidual = residual - dot(x, residual) .* x;
end