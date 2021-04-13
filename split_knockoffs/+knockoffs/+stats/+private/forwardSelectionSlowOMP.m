function varargout = forwardSelectionSlowOMP(X, y)
% KNOCKOFFS.STATS.PRIVATE.FORWARDSELECTIONSLOWOMP  Slow reference implementation of forward 
%   selection with orthogonal matching pursuit (OMP)
%
% See also KNOCKOFFS.STATS.PRIVATE.FORWARDSELECTIONOMP

function residual = target(X, x, ~)
    warning_state = warning('off', 'stats:regress:RankDefDesignMat');
    [~,~,residual] = regress(y, [X x]);
    warning(warning_state);
end

[varargout{1:nargout}] = ...
    knockoffs.stats.private.sequentialfs(@criterion, @target, X, y);

end

function c = criterion(~, x, residual)
    c = -abs(dot(x, residual));
end