function X_ko = fixed_MinCondCov(X, randomize)
% KNOCKOFFS.CREATE.FIXED_MINCONDCOV  Create knockoff variables using a SDP
%   X_ko = KNOCKOFFS.CREATE.FIXED_MINCONDCOV(X) create knockoffs deterministically
%   X_ko = KNOCKOFFS.CREATE.FIXED_MINCONDCOV(X, true) knockoffs with randomization
%
%   Creates knockoff variables using semi-definite programming (SDP) to
%   minimize conditional covariances.
%
%   Inputs:
%       X - n x p scaled data matrix (n >= 2*p)
%       randomize - whether to use randomization in the construction of
%                   the knockoff variables
%   
%   Outputs:
%       X_ko - n x p knockoff variable matrix
%
%   See also KNOCKOFFS.CREATE.FIXED_EQUI, KNOCKOFFS.CREATE.FIXED_SDP.

% Validate inputs.
if ~exist('randomize', 'var'), randomize = []; end

% Validate CVX version.
% (Build 1079 was released on Apr 23, 2014).
if ~exist('cvx_begin', 'file')
    error('knockoff:MissingCVX', ...
          'CVX is not installed. To use SDP knockoffs, please install CVX.')
elseif knockoffs.private.cvxBuild() < 1079
    error('knockoff:OldCVX', ...
          'CVX is too old. To use MinCondCov knockoffs, please upgrade CVX.')
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
         'Model is not identifiable, but proceeding with MinCondCov knockoffs.'])
    S_inv(S_zeros) = 0;
end
S_inv = diag(S_inv);

% Compute the Gram matrix X'*X, conditional covariances, and its (pseudo)inverse.
G = V * sparse(S.^2) * V';
sc = nan(size(G,1),1);
for i = 1:size(G,1)
  noti = setdiff(1:size(G,1),i);
  G11 = G(i,i)*[1 1;1 1];
  G12 = repmat(G(i,noti),2,1);
  G22 = G(noti,noti);
  SC = G11 - G12*(G22\G12');
  sc(i) = SC(1,2);
end
G_inv = V * sparse(S_inv.^2) * V';

% Optimize the parameter s of Equation 1.3 according to the SDP
% minimization problem of Equation 2.5.
s = solveMinCondCov(G,sc);
s(s <= tol) = 0;
diag_s = sparse(diag(s));

% Construct the knockoff according to Equation 1.4:
%   X_ko = X(I - (X'X)^{-1} * s) + U_perp * C
% where
%   C'C = 2s - s * (X'X)^{-1} * s.
[~,D,V] = knockoffs.private.canonicalSVD(2*diag_s - diag_s*G_inv*diag_s);
d = sqrt(max(0, diag(D)));
diag_d = sparse(diag(d));
X_ko = X - X * G_inv * diag_s + U_perp * diag_d * V';

end

function y = solveMinCondCov(G,sc) %#ok<STOUT>
% Solves 
%
% maximize    |y-sc|
% subect to   s <= 1
%             [G , G - diag(s); G - diag(s) , G] >= 0
%
% The LMI is equivalent to 2G - diag(s) >= 0 and s >= 0
% 
% Using CVX, we solve this via the dual SDP.

p = length(G);

cvx_begin quiet
    variable y(p);
    minimize(norm(y-sc,1))
    2*G-diag(y) == semidefinite(p); %#ok<EQEFF>
    0 <= y <= 1 %#ok<CHAIN,NOPRT>
cvx_end

end