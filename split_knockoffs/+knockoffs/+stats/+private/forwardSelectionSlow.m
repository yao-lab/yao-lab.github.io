function varargout = forwardSelectionSlow(X, y)
% KNOCKOFFS.STATS.PRIVATE.FORWARDSELECTIONSLOW  Slow reference implementation of forward selection
%
% See also KNOCKOFFS.STATS.PRIVATE.FORWARDSELECTION

[varargout{1:nargout}] = ...
    knockoffs.stats.private.sequentialfs(@criterion, @target, X, y);

end

function c = criterion(~, x, residual)
    c = -abs(dot(x, residual));
end

function nextResidual = target(~, x, residual)
    [~,~,nextResidual] = regress(residual, x);
end