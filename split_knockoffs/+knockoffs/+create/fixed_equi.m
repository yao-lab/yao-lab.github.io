function X_ko = fixed_equi(X, randomize)
% KNOCKOFFS.CREATE.FIXED_EQUI  Create equi-correlated knockoff variables for a fixed design
%   X_ko = KNOCKOFFS.CREATE.FIXED_EQUI(X) create knockoffs deterministically
%   X_ko = KNOCKOFFS.CREATE.FIXED_EQUI(X, true) create knockoffs with randomization
%
%   Inputs:
%       X - n x p scaled data matrix (n >= 2*p)
%       randomize - whether to use randomization in the construction of
%                   the knockoff variables
%   
%   Outputs:
%       X_ko - n x p knockoff variable matrix
%
%   See also KNOCKOFFS.FIXED.CREATESDP, KNOCKOFFS.FIXED.CREATEMINCONDCOV.

if ~exist('randomize', 'var'), randomize = []; end

% Compute SVD and U_perp.
[U,S,V,U_perp] = knockoffs.private.decompose(X, randomize);

% Set s = min(2 * smallest eigenvalue of X'X, 1), so that all the
% correlations have the same value X_j'X_j = 1 - s.
if any(diag(S) <= 1e-5 * max(diag(S)))
    error('knockoff:RankDeficiency', ...
          ['Data matrix is rank deficient. ' ...
           'Equicorrelated knockoffs will have no power. ' ...
           'If you must proceed, use SDP knockoffs instead.'])
end
lambda_min = min(diag(S))^2;
s = min(2*lambda_min, 1);

% Construct the knockoff according to Equation 1.4:
%   X_ko = X(I - (X'X)^{-1} * s) + U_perp * C
% where
%   C'C = 2s - s * (X'X)^{-1} * s.
p = size(X, 2);
U_perp = U_perp(:, 1:p);
X_ko = U * sparse(diag(diag(S) - s./diag(S))) * V' + ...
    U_perp * sparse(diag(sqrt(2*s - s^2./diag(S).^2))) * V';
X_ko = real(X_ko);

end