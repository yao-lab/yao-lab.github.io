% Generate some random data
n = 100;
data = rand(n,2);

% Compute distance matrix
dist = pdist(data);

% Initialize clusters
clusters = num2cell(1:n);

% Merge clusters until there is only one left
while numel(clusters) > 1
    % Find closest pair of clusters
    [min_dist, idx] = min(dist);
    [i,j] = ind2sub([numel(clusters), numel(clusters)], idx);
    
    % Merge clusters i and j
    clusters{i} = [clusters{i} clusters{j}];
    clusters(j) = [];
    
    % Update distance matrix
    dist(:,j) = [];
    dist(j,:) = [];
    dist(i,:) = min(dist(i,:), dist(j,:));
    dist(:,i) = dist(i,:)';
end

% Compute 0-dimensional persistent homology
stream = api.Plex4.createExplicitSimplexStream();
for i = 1:n
    stream.addVertex(i,0);
end
for i = 1:numel(clusters)
    stream.addElement(clusters{i}-1,1);
end
stream.finalizeStream();
persistence = api.Plex4.getModularSimplicialAlgorithm(1,2);
intervals = persistence.computeIntervals(stream);

% Plot barcode
options.filename = 'barcode.png';
options.max_filtration_value = 1;
plot_barcodes(intervals, options);