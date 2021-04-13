function X_k = gaussian(X, mu, Sigma, method)
%KNOCKOFFS.CREATE.GAUSSIAN Samples model-free multivariate normal knockoffs according
%  to the classical regression formulas, after computing the diagonal
%  matrix 'diag_s' according to the specified method.
%
%  X_k = KNOCKOFFS.CREATE.GAUSSIAN(X, mu, Sigma)
%  X_k = KNOCKOFFS.CREATE.GAUSSIAN(X, mu, Sigma, method)
%
%  Inputs:
%       X    - n x p scaled data matrix
%       mu   - 1 x p mean vector for the marginal distribution of X
%     Sigma  - p x p covariance matrix for the marginal distribution of X
%     diag_s - p x p diagonal matrix (equation 3.2)
%     method - either 'equi' (for equi-correlated knockoffs), 'sdp'
%              (for knockoffs optimized using semi-definite programming) or 
%              'asdp' (for approximate SDP knockoffs)
%              Default: 'sdp'
%  Outputs:
%       X_k - n x p matrix of knockoff variables
%
%  See also KNOCKOFFS.CREATE.GAUSSIAN_SAMPLE, KNOCKOFFS.CREATE.FIXED

if ~exist('method', 'var') || isempty(method), method = 'sdp'; end;

% Compute the diagonal matrix diag_s
method = lower(method);
switch method
    case 'equi'
        diag_s = sparse(diag(knockoffs.create.solveEqui(Sigma)));
    case 'sdp'
        diag_s = sparse(diag(knockoffs.create.solveSDP(Sigma)));
    case 'asdp'
        diag_s = sparse(diag(knockoffs.create.solveASDP(Sigma)));    
    otherwise
        error('Invalid Model-X Gaussian knockoff creation method %s', method)
end

% Sample the knockoffs
X_k = knockoffs.create.gaussian_sample(X, mu, Sigma, diag_s);

end