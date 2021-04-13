function X_k = gaussian_sample(X, mu, Sigma, diag_s)
%KNOCKOFFS.GAUSSIAN.SAMPLE Samples model-free multivariate normal knockoffs according
%  to the classical regression formulas, for a pre-computed diagonal matrix
%  'diag_s'.
%
%   X_k = KNOCKOFFS.CREATE.GAUSSIAN_SAMPLE(X, mu, Sigma, diag_s)
%
%   Inputs:
%       X    - n x p scaled data matrix
%       mu   - 1 x p mean vector for the marginal distribution of X
%     Sigma  - p x p covariance matrix for the marginal distribution of X
%     diag_s - p x p diagonal matrix (equation 3.2)
%   
%   Outputs:
%       X_k - n x p matrix of knockoff variables
%
%   See also KNOCKOFFS.CREATE.GAUSSIAN

[n,p] = size(X);

% Compute the inverse covariance matrix of the original variables and 
% multiply it by the diagonal matrix diag_s
SigmaInv_s = Sigma\diag_s;

% Compute mean and covariance of the knockoffs
mu_k = X-(X-repmat(mu,n,1))*SigmaInv_s;
Sigma_k = 2*diag_s - diag_s*SigmaInv_s;

% Sample the knockoffs
X_k = mu_k + randn(n,p)*chol(Sigma_k);

end

