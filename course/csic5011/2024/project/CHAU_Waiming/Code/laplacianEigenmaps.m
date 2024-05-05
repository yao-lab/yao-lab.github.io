function [ Y, output ] = laplacianEigenmaps( X, varargin )
%LAPLACIANEIGENMAPS Nonlinear dimensionality reduction using Laplacian
%eigenmaps.
%   Y = laplacianEigenmaps(X) returns the embedding of the N by D matrix X
%   into two dimensions. Each row of X represents an observation.
%
%   [Y, output] = laplacianEigenmaps(X) returns a structure containing
%   details of the embedding.
%
%   [...] = laplacianEigenmaps(..., 'param1',val1, 'param2',val2, ...)
%   specifies optional parameter name/value pairs to control further
%   details of the embedding.
%
%   Parameters are:
%   'Distance' - A string specifies the distance metric used to calculate
%                    distance between observations. Default: 'euclidean' 
%
%   'NumNeighbors' - A positive integer specifying the number of neighbors
%                    to consider in the adjacency matrix. Default: 10
%
%   'NumDimensions'- A positive integer specifying the number of dimension
%                    of the representation Y. Default: 2
%
%   'Sigma' - A real scalar speciying the standard deviation of the
%                    Gaussian heat kernel used. Default: 1
%
%   'Verbose' - Controls the level of detail of command line display. 
%                       Default: 1.
%                       0: Do not display anything
%                       1: Display summary information and timing after
%                          different algorithm stages.
%
%
%   References:
%       [1] Belkin, Mikhail, and Partha Niyogi. "Laplacian eigenmaps for
%           dimensionality reduction and data representation." Neural
%           computation 15.6 (2003): 1373-1396.
%
%       [2] Bengio, Yoshua, et al. "Learning eigenfunctions links spectral
%           embedding and kernel PCA." Learning 16.10 (2006).
%
%       [3] Ng, Andrew Y., Michael I. Jordan, and Yair Weiss. "On spectral
%           clustering: Analysis and an algorithm." Advances in neural
%           information processing systems. 2002.
%
%       [4] Van Der Maaten, Laurens, Eric Postma, and Jaap Van den Herik.
%           "Dimensionality reduction: a comparative review." J Mach Learn
%           Res 10 (2009): 66-71.
%
%
%   Example:
%       N = 10000; noise = 0.05;
%       t = 3*pi/2 * (1 + 2*rand(N,1));
%       h = 11 * rand(N,1);
%       X = [t.*cos(t), h, t.*sin(t)] + noise*randn(N,3);
%       Y = laplacianEigenmaps( X );
%       % Plot 3D input and 2D result
%       figure('Position',[200,500,1000,1000],'WindowStyle','docked');
%       subplot(1,2,1); title('Swiss roll');
%       scatter3(X(:,1),X(:,2),X(:,3),10,t,'fill');
%       subplot(1,2,2); title('Embedding');
%       scatter(Y(:,1),Y(:,2),10,t,'fill');
%
%
%  (c) 2017 Jacob Zavatone-Veth (MIT License)
%

%% Parse arguments

% Get size of input matrix
if ~ismatrix(X) || ~isnumeric(X) || ~isreal(X)
    error('LaplacianEigenmaps:Data array must be a real numeric matrix.');
end
[n, d] = size(X);

% Check number of output arguments
if nargout > 2
    error('LaplacianEigenmaps:Invalid number of output arguments.');
end

% Parse other inputs
paramNames = {'Distance', 'NumNeighbors', 'NumDimensions', 'Sigma', 'Verbose'};
defaults   = {'euclidean',	10, 2, 1, 1};
[metric, nneighbors, ydims, kernelstddev, verbose] =...
    internal.stats.parseArgs(paramNames, defaults, varargin{:});

% Validate inputs
if ydims > d
    error('LaplacianEigenmaps:Invalid output dimensionality.');
end
if ~any(verbose==[0 1])
    error('LaplacianEigenmaps:Invalid verbosity');
end

% Allocate a container to store self-timing information
elapsed = zeros(4,1);

%% Form the graph
tic;

% Perform KNN search
[ind, A] = knnsearch(X, X, 'K', nneighbors + 1, 'Distance', metric);

% Form the adjacency matrix
A = sparse(repmat((1:n)', 1, nneighbors+1), ind, A, n, n);

% Symmetrize the adjacency matrix to form an undirected graph
A = max(A, A');

% Scale the adjacency matrix so the maximal distance is 1
A = A.^2;
max_distance = max(nonzeros(A));
A = A./max_distance;

% Build an undirected graph from the adjacency matrix
G = graph(A);

% Find the connected components of the graph
[bins] = conncomp(G, 'OutputForm', 'cell');

% If there are multiple connected components, discard all but the largest
if length(bins) > 1
    fprintf('Discarding %d connected components.\n', length(bins)-1);

    % Find which connected component is the largest
    [~, idx] = max(cellfun(@nnz, bins));

    % Discard all other connected components
    inds = 1:length(bins);
    inds = inds(inds~=idx);
    for ind = inds
        A(bins{ind}, bins{ind}) = 0;
    end
end

elapsed(2) = toc;
if verbose>0
    fprintf('Constructed graph from input distances in %f seconds.\n', elapsed(2));
end

%% Compute the graph Laplacian
tic;

% Evaluate Gaussian kernel on nonzero elements of the adjacency matrix
A = spfun(@(x) exp(-x / (2 * kernelstddev ^ 2)), A);

% Construct the diagonal degree matrix
D = diag(sum(A, 2));

% Compute the (unnormalized) graph Laplacian
L = D - A;

% Remove nonsense values
L(isnan(L) | isinf(L)) = 0;
D(isnan(D) | isinf(D)) = 0;

elapsed(3) = toc;
if verbose>0
    fprintf('Computed graph Laplacian in %f seconds.\n', elapsed(3));
end

%% Eigendecompose the graph Laplacian
tic;

% Solve the generalize eigenvalue problem using ARPACK
% Note that we only need the eigenvectors corresponding to the smallest
% ydims+1 eigenvalues
[Y, eigenvalues, exitFlag] = eigs(L, D, ydims + 1,'smallestabs','Display', verbose);
if exitFlag ~= 0
    error('ARPACK solver failed to converge.');
end

% Sort eigenvectors in ascending order
eigenvalues = diag(eigenvalues);
[eigenvalues, ind] = sort(eigenvalues, 'ascend');
eigenvalues = eigenvalues(2:ydims + 1);

% Get the final embedding
Y = Y(:,ind(2:ydims + 1));

elapsed(4) = toc;
if verbose>0
    fprintf('Eigendecomposed graph Laplacian in %f seconds.\n', elapsed(4));
end

%% Store mapping details in structure
if nargout == 2
    output = struct(...
        'distance', metric,...
        'numneigbors', nneighbors,...
        'numdimensions', ydims,...
        'sigma', kernelstddev,...
        'max_distance', max_distance,...
        'graph', A,...
        'mapped_data', Y,...
        'eigenvalues', eigenvalues...
        );
end

if verbose>0
    fprintf('Mapping completed in %f seconds.\n\n', sum(elapsed));
end

end