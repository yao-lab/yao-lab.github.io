function [added, history] = sequentialfs(crit_fn, target_fn, X, y)
% KNOCKOFFS.STATS.PRIVATE.SEQUENTIALFS  Sequential feature selection
%   added = KNOCKOFFS.STATS.PRIVATE.SEQUENTIALFS(crit_fn, target_fn, X, y, ...)
%   [added, history] = KNOCKOFFS.STATS.PRIVATE.SEQUENTIALFS(crit_fn, target_fn, X, y, ...)
%
%   This function is a variant of the standard MATLAB function of the same
%   name. It omits many features of that function, but it adds the ability
%   to change the target function at every step. This is useful for
%   computing residuals.

[n,p] = size(X);
assert(isequal(size(y), [n 1]));

added = zeros(1,p);
in = false(1,p);
target = y;

if nargout > 1
    history = struct('Crit', zeros(1,p), ...
                     'Target', zeros(p,n), ...
                     'In', false(p,p));
    history.Target(1,:) = target;
end

for step = 1:p
    X_in = X(:,in);
    available = find(~in);

    % Find the best variable to add among the remaining variables.
    criteria = zeros(1, length(available));
    for j = 1:length(available)        
        criteria(j) = crit_fn(X_in, X(:,available(j)), target);
    end
    [best_crit, best_j] = min(criteria);
    best_var = available(best_j);
    added(step) = best_var;
    in(best_var) = true;
    
    % Compute the new target from the old.
    if step ~= p
        target = target_fn(X_in, X(:,best_var), target);
    end
    
    % Update history, if necessary.
    if nargout > 1
        history.In(step,:) = in;
        history.Crit(step) = best_crit;
        if step ~= p
            history.Target(step+1,:) = target;
        end
    end
    
end