function s = solveEqui(Sigma)
%KNOCKOFFS.CREATE.EQUI Computes the diagonal matrix 'diags_s', used to sample
% Model-X and Fixed-X equi-correlated knockoffs.
%
%   s = KNOCKOFFS.CREATE.SOLVEEQUI(Sigma)
%
%  Inputs:
%     Sigma  - p x p covariance matrix for the marginal distribution of X
%
%  Outputs:
%          s - a vector of length p
%
% See also KNOCKOFFS.CREATE.SOLVESDP, KNOCKOFFS.CREATE.SOLVEASDP,
% KNOCKOFFS.CREATE.GAUSSIAN, KNOCKOFFS.CREATE.FIXED

% Convert the covariance matrix into a correlation matrix
[scaleSigma, corrSigma] = cov2corr(Sigma);

opts.isreal = true;
opts.tol = 1e-6;
lambda_min = eigs(corrSigma,1,'sm',opts);
s = ones(length(corrSigma),1) * min(2*lambda_min, min(diag(corrSigma)));

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
s = s-s_eps/10;
s(s < 0) = 0;

% Scale back the results for a covariance matrix
s = s(:) .* (scaleSigma(:).^2);

end