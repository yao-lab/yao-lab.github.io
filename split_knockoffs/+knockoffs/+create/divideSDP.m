function [clusters_sdp,subSigma] = divideSDP(Sigma)
%KNOCKOFFS.CREATE.DIVIDESDP Approximate a covariance matrix by a block diagonal matrix,
% in order to efficiently construct approximate SDP knockoffs.
%
% This function is used to create approximate SDP knockoffs.
% The full covariance matrix is approximated by a block diagonal matrix
% constructed by clustering the columns of the original covariance matrix.
% The clusters are created from a single linkage dendrogram by joining the 
% leaves greedely in such a way that no cluster contains more than 10
% percent of all variables.
%
%   [,subSigma] = KNOCKOFFS.CREATE.DIVIDESDP(Sigma)
%
%  Inputs:
%     Sigma  - p x p covariance matrix for the marginal distribution of X
%
%  Outputs:
%    subSigma - a cell array of smaller covariance matrices
%
% See also KNOCKOFFS.CREATE.GAUSSIAN_ASDP, KNOCKOFFS.CREATE.GAUSSIAN_SDP

p = length(Sigma);

% Parameters for the max-size clustering algorithm
linkmeth = 'single'; %average, complete
maxclust = floor(p/10);

% Compute the clustering dendrogram
Z = linkage(1-abs(Sigma(tril(true(p),-1)))',linkmeth);

% Create clusters adaptively from the dendrogram, making sure that no
% cluster contains more than 'maxclust' elements
clusters_sdp = (1:p)';
clustersizes = zeros(max(max(Z(:,1:2))),1);
clustersizes(1:p) = 1;
for j = 1:size(Z,1)
  if sum(clustersizes(Z(j,1:2)))<=maxclust
    clusters_sdp(ismember(clusters_sdp,Z(j,1:2))) = p+j;
    clustersizes(p+j) = sum(clustersizes(Z(j,1:2)));
    clustersizes(Z(j,1:2)) = 0;
  end
end
uclusters_sdp = unique(clusters_sdp);
nclust_sdp = length(uclusters_sdp);

% Create the block matrices and rename the unique clusters
maxsubp = 0;
subSigma = cell(1,nclust_sdp);
for ksub = 1:nclust_sdp
  k = uclusters_sdp(ksub);
  k_indices = clusters_sdp==k;
  subp = sum(k_indices);
  maxsubp = max(maxsubp,subp);
  subSigma{ksub} = Sigma(k_indices,k_indices);
  clusters_sdp(k_indices) = ksub;
end

end