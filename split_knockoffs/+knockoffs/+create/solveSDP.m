function s = solveSDP(Sigma)
%KNOCKOFFS.CREATE.SOLVESDP Computes the diagonal matrix 'diags_s', used to sample
% Model-X and Fixed-X SDP knockoffs.
%    
% Solves
%
% maximize    1' * s
% subect to   0 <= s_i <= 1, (for all 1<=i<=p)
%             G = [Sigma , Sigma - diag(s1); Sigma - diag(s1) , Sigma] >= 0
%
% The LMI is equivalent to 2G - diag(s) >= 0 and s >= 0
% 
% Using CVX, we solve this via the dual SDP.
%
%   s = KNOCKOFFS.CREATE.SOLVESDP(Sigma)
%
%  Inputs:
%     Sigma  - p x p covariance matrix for the marginal distribution of X
%
%  Outputs:
%          s - a vector of length p
%
% See also KNOCKOFFS.CREATE.SOLVEEQUI, KNOCKOFFS.CREATE.SOLVEASDP,
% KNOCKOFFS.CREATE.GAUSSIAN, KNOCKOFFS.CREATE.FIXED


% Validate CVX version.
% (Build 1079 was released on Apr 23, 2014).
if ~exist('cvx_begin', 'file')
    error('knockoff:MissingCVX', ...
          'CVX is not installed. To use SDP knockoffs, please install CVX.')
elseif knockoffs.private.cvxBuild() < 1079
    error('knockoff:OldCVX', ...
          'CVX is too old. To use SDP knockoffs, please upgrade CVX.')
end

p = length(Sigma);

% Convert the covariance matrix into a correlation matrix
[scaleSigma, corrSigma] = cov2corr(Sigma);

% Optimize the parameter s of Equation 3.2 according to the SDP
% minimization problem of Equation 2.14.
warning('off')
cvx_begin quiet
    variable s(p);
    maximize(sum(s)) %#ok<NODEF>
    2*corrSigma-diag(s) == semidefinite(p); %#ok<EQEFF>
    0 <= s <= 1 %#ok<CHAIN,NOPRT>
cvx_end
warning('on')
s(s < 0) = 0;

% Try different solver if the first one fails
if (any(isnan(s)))
    error('CVX failed to solve the SDP required to construct knockoffs. Trying again with a different solver. To hange the current solver, type: cvx_solver <solver name>');
end

% Compensate for numerical errors in CVX
psd = 1;
s_eps = 1e-8;
while psd~=0
    % Compute knockoff conditional covariance matrix
    diag_s = sparse(diag(s.*(1-s_eps)));
    SigmaInv_s = corrSigma\diag_s;
    Sigma_k = 2*diag_s - diag_s*SigmaInv_s;

    [~,psd] = chol(Sigma_k);
    s_eps = s_eps*10;
end
s = s.*(1-s_eps/10);
s(s < 0) = 0;

% Scale back the results for a covariance matrix
s = s(:) .* (scaleSigma(:).^2);

end

