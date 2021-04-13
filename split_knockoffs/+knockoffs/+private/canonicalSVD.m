function [U,S,V] = canonicalSVD(X)
% KNOCKOFFS.PRIVATE.CANONICALSVD  Reduced SVD with canonical sign choice
%   [U,S,V] = KNOCKOFFS.PRIVATE.CANONICALSVD(X)
%
%   Computes a reduced SVD without sign ambiguity. Our convention is that
%   the sign of each vector in U is chosen such that the coefficient
%   with largest absolute value is positive.

[U,S,V] = svd(X,0);

for j = 1:min(size(X))
    [~,i] = max(abs(U(:,j)));
    if U(i,j) < 0
        U(:,j) = -U(:,j);
        V(:,j) = -V(:,j);
    end
end

end