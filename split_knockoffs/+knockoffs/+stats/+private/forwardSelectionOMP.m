function varargout = forwardSelectionOMP(X, y)
% KNOCKOFFS.STATS.PRIVATE.FORWARDSELECTIONOMP  Fast implementation of 
%  forward selection with orthogonal mathcing pursuit (OMP)
%
%   Assumes that the columns of X are normalized to 1.
%
% See also KNOCKOFFS.STATS.PRIVATE.FORWARDSELECTION

[n,p] = size(X);

X = knockoffs.private.normc(X);  % Standardize the variables

Q = zeros(n,p);
i = 1;

function nextResidual = target(~, x, residual)
    % Orthonormalize using modified Gram-Schmidt.
    for j = 1:i-1
        x = x - dot(Q(:,j), x) .* Q(:,j);
    end
    q = x / norm(x);
    
    nextResidual = residual - dot(q,y) .* q;
    Q(:,i) = q;
    i = i+1;
end

[varargout{1:nargout}] = ...
    knockoffs.stats.private.sequentialfs(@criterion, @target, X, y);

end

function c = criterion(~, x, residual)
    c = -abs(dot(x, residual));
end