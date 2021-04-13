function s = solveASDP(Sigma, parallel)
%KNOCKOFFS.CREATE.SOLVEASDP Computes the diagonal matrix 'diags_s', used to 
% create Model-X or Fixed-X approximate SDP knockoffs.
%
% This function approximates the covariance matrix Sigma with a
% block-diagonal matrix constructed by clustering its columns. 
% The clusters are created from a single linkage dendrogram by joining the 
% leaves greedely in such a way that no cluster contains more than 10
% percent of all variables.
%
% Then, the approximate SDP problem factors into a number of independent 
% subproblems that can be efficiently solved in parallel.
%
%   s = KNOCKOFFS.CREATE.SOLVEASDP(Sigma)
%   s = KNOCKOFFS.CREATE.SOLVEASDP(Sigma, parallel)
%
%  Inputs:
%     Sigma  - p x p covariance matrix for the marginal distribution of X
%   parallel - whether to solve the subproblems in parallel (default: false)
%
%  Outputs:
%    s       - a vector of length p
%
% See also KNOCKOFFS.CREATE.SOLVESDP, KNOCKOFFS.CREATE.SOLVEEQUI, 
% KNOCKOFFS.CREATE.DIVIDESDP, KNOCKOFFS.CREATE.GAUSSIAN, KNOCKOFFS.CREATE.FIXED

p = length(Sigma);
if ~exist('parallel', 'var') || isempty(parallel), parallel = false; end;


% Approximate the covariance matrix with a block diagonal matrix
[clusters_sdp,subSigma] = knockoffs.create.divideSDP(Sigma);

% Create the smaller SDP problems and solve them in parallel
nclust_sdp = length(subSigma);
sub_diag_s = cell(1,nclust_sdp);
if parallel
    parfor ksub = 1:nclust_sdp
        sub_diag_s{ksub} = knockoffs.create.solveSDP(subSigma{ksub});
    end
else
    for ksub = 1:nclust_sdp
        sub_diag_s{ksub} = knockoffs.create.solveSDP(subSigma{ksub});
    end
end

% Put the results to all subproblems back together
s_asdp = nan(p,1);
for ksub = 1:nclust_sdp
    s_asdp(clusters_sdp==ksub) = sub_diag_s{ksub};
end

% Find the optimal shrinking factor to ensure positive-definiteness
% via binary search
iterations = 20;
gamma_sdp = 1/2;
for j = 2:iterations
  [~,psd] = chol(2*Sigma-gamma_sdp*diag(s_asdp));
  if psd==0
    gamma_sdp = gamma_sdp + 1/2^j;
  else
    gamma_sdp = gamma_sdp - 1/2^j;
  end
end
[~,psd]=chol(2*Sigma-gamma_sdp*diag(s_asdp));
if psd~=0
    gamma_sdp = gamma_sdp - 1/2^j;
end

% Shrink the solution
s = gamma_sdp*s_asdp;

end