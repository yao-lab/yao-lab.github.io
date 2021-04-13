function X_ko = fixed_SDP(X, randomize, approximate)
% KNOCKOFFS.CREATE.FIXED_SDP  Create knockoff variables using SDP
%   X_ko = KNOCKOFFS.CREATE.FIXED_SDP(X) create knockoffs deterministically
%   X_ko = KNOCKOFFS.CREATE.FIXED_SDP(X, true) create knockoffs with randomization
%   X_ko = KNOCKOFFS.CREATE.FIXED_SDP(X, false, true) create approximate SDP knockoffs
%
%   Creates knockoff variables using semi-definite programming (SDP).
%
%   Inputs:
%       X         - n x p scaled data matrix (n >= 2*p)
%       randomize - whether to use randomization in the construction of
%                   the knockoff variables (default: false)
%     approximate - whether to use approximate sdp construction 
%                   (default: false)
%   
%   Outputs:
%       X_ko - n x p knockoff variable matrix
%
%   See also KNOCKOFFS.CREATE.FIXED_EQUI, KNOCKOFFS.CREATE.FIXED_MINCONDCOV.

% Validate inputs.
if ~exist('randomize', 'var'), randomize = []; end
if ~exist('approximate', 'var'), approximate = false; end

% Validate CVX version.
% (Build 1079 was released on Apr 23, 2014).
if ~exist('cvx_begin', 'file')
    error('knockoff:MissingCVX', ...
          'CVX is not installed. To use SDP knockoffs, please install CVX.')
elseif knockoffs.private.cvxBuild() < 1079
    error('knockoff:OldCVX', ...
          'CVX is too old. To use SDP knockoffs, please upgrade CVX.')
end

% Compute SVD and U_perp.
[~,S,V,U_perp] = knockoffs.private.decompose(X, randomize);

% Check for rank deficiency.
tol = 1e-5;
S_inv = 1 ./ diag(S);
S_zeros = diag(S) <= tol*max(diag(S));
if any(S_zeros)
    warning('knockoff:RankDeficiency', ...
        ['Data matrix is rank deficient. ' ...
         'Model is not identifiable, but proceeding with SDP knockoffs.'])
    S_inv(S_zeros) = 0;
end
S_inv = diag(S_inv);

% Compute the Gram matrix X'*X and its (pseudo)inverse.
G = V * sparse(S.^2) * V';
G_inv = V * sparse(S_inv.^2) * V';

% Optimize the parameter s of Equation 1.3 according to the SDP
% minimization problem of Equation 2.5.
if approximate
    diag_s = sparse(diag(knockoffs.create.solveASDP(G)));
else
    diag_s = sparse(diag(knockoffs.create.solveSDP(G)));
end

% Construct the knockoff according to Equation 1.4:
%   X_ko = X(I - (X'X)^{-1} * s) + U_perp * C
% where
%   C'C = 2s - s * (X'X)^{-1} * s.
[~,D,V] = knockoffs.private.canonicalSVD(2*diag_s - diag_s*G_inv*diag_s);
d = sqrt(max(0, diag(D)));
diag_d = sparse(diag(d));
p = size(X, 2);
X_ko = X - X * G_inv * diag_s + U_perp(:, 1:p) * diag_d * V';

end