function [mappedX, mapping] = Isomap(X, no_dims, k)
%ISOMAP Runs the Isomap algorithm
%
%   [mappedX, mapping] = isomap(X, no_dims, k); 
%
% The functions runs the Isomap algorithm on dataset X to reduce the
% dimensionality of the dataset to no_dims. The number of neighbors used in
% the compuations is set by k (default = 12). This implementation does not
% use the Landmark-Isomap algorithm.
%
% If the neighborhood graph that is constructed is not completely
% connected, only the largest connected component is embedded. The indices
% of this component are returned in mapping.conn_comp.
%
%
% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology
    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
    if ~exist('k', 'var')
        k = 12;
    end
    % Construct neighborhood graph
    disp('Constructing neighborhood graph...'); 
    D = real(find_nn(X, k));
    
    % Select largest connected component
    blocks = components(D)';
    count = zeros(1, max(blocks));
    for i=1:max(blocks)
        count(i) = length(find(blocks == i));
    end
    [count, block_no] = max(count);
    conn_comp = find(blocks == block_no);    
    D = D(conn_comp, conn_comp);
    mapping.D = D;
    n = size(D, 1);
    % Compute shortest paths
    disp('Computing shortest paths...');
    D = dijkstra(D, 1:n);
    mapping.DD = D;
    
    % Performing MDS using eigenvector implementation
    disp('Constructing low-dimensional embedding...');
    D = D .^ 2;
    M = -.5 .* (bsxfun(@minus, bsxfun(@minus, D, sum(D, 1)' ./ n), sum(D, 1) ./ n) + sum(D(:)) ./ (n .^ 2));
    M(isnan(M)) = 0;
    M(isinf(M)) = 0;
    [vec, val] = eig(M);
	if size(vec, 2) < no_dims
		no_dims = size(vec, 2);
		warning(['Target dimensionality reduced to ' num2str(no_dims) '...']);
	end
	
    % Computing final embedding
    [val, ind] = sort(real(diag(val)), 'descend'); 
    vec = vec(:,ind(1:no_dims));
    val = val(1:no_dims);
    mappedX = real(bsxfun(@times, vec, sqrt(val)'));
    
    % Store data for out-of-sample extension
    mapping.conn_comp = conn_comp;
    mapping.k = k;
    mapping.X = X(conn_comp,:);
    mapping.vec = vec;
    mapping.val = val;
    mapping.no_dims = no_dims;
    
    
    function [D, ni] = find_nn(X, k)
%FIND_NN Finds k nearest neigbors for all datapoints in the dataset
%
%	[D, ni] = find_nn(X, k)
%
% Finds the k nearest neighbors for all datapoints in the dataset X.
% In X, rows correspond to the observations and columns to the
% dimensions. The value of k is the number of neighbors that is
% stored. The function returns a sparse distance matrix D, in which
% only the distances to the k nearest neighbors are stored. For
% equal datapoints, the distance is set to a tolerance value.
% The method is relatively slow, but has a memory requirement of O(nk).
%
%
% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology
	if ~exist('k', 'var') || isempty(k)
		k = 12;
    end
    
    % Perform adaptive neighborhood selection if desired
    if ischar(k)
        [D, max_k] = find_nn_adaptive(X);
        ni = zeros(size(X, 1), max_k);
        for i=1:size(X, 1)
            tmp = find(D(i,:) ~= 0);
            tmp(tmp == i) = [];
            tmp = [tmp(2:end) zeros(1, max_k - length(tmp) + 1)];
            ni(i,:) = tmp;
        end
    
    % Perform normal neighborhood selection
    else
        
        % Compute distances in batches
        n = size(X, 1);
        sum_X = sum(X .^ 2, 2);
        batch_size = round(2e7 ./ n);
        D = zeros(n, k);
        ni = zeros(n, k);
        for i=1:batch_size:n
            batch_ind = i:min(i + batch_size - 1, n);
            DD = bsxfun(@plus, sum_X', bsxfun(@plus, sum_X(batch_ind), ...
                                                   -2 * (X(batch_ind,:) * X')));
            [DD, ind] = sort(abs(DD), 2, 'ascend');
            D(batch_ind,:) = sqrt(DD(:,2:k + 1));
            ni(batch_ind,:) = ind(:,2:k + 1);
        end
        D(D == 0) = 1e-9;
        Dout = sparse(n, n);
        idx = repmat(1:n, [1 k])';
        Dout(sub2ind([n, n], idx,   ni(:))) = D;
        Dout(sub2ind([n, n], ni(:), idx))   = D;
        D = Dout;
    end
    function blocks = components(A)
%COMPONENTS Finds connected components in a graph defined by a adjacency matrix
%
%   blocks = components(A)
%
% Finds connected components in a graph defined by the adjacency matrix A.
% The function outputs an n-vector of integers 1:k in blocks, meaning that
% A has k components. The vector blocks labels the vertices of A according 
% to component.
% If the adjacency matrix A is undirected (i.e. symmetric), the blocks are 
% its connected components. If the adjacency matrix A is directed (i.e. 
% unsymmetric), the blocks are its strongly connected components.
%
%
% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology
    % Check size of adjacency matrix
    [n, m] = size(A);
    if n ~= m, error ('Adjacency matrix must be square'), end;
    % Compute Dulmage-Mendelsohn permutation on A
    if ~all(diag(A)) 
        [foo, p, bar, r] = dmperm(A | speye(size(A)));
    else
        [foo, p, bar, r] = dmperm(A);  
    end
    % Compute sizes and number of clusters
    sizes = diff(r);
    k = length(sizes);
    % Now compute the array blocks
    blocks = zeros(1, n);
    blocks(r(1:k)) = ones(1, k);
    blocks = cumsum(blocks);
    % Permute blocks so it maps vertices of A to components
    blocks(p) = blocks;
    