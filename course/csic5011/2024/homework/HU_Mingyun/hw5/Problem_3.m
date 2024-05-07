% Set up the problem parameters
l = 10; % sequence length
theta = 5; % threshold distance
sigma = 0.1; % noise standard deviation

% Generate random 3D coordinates for the amino acids
X_true = rand(l, 3);

% Compute the true pairwise distances
D_true = pdist(X_true);

% Compute the contact map graph
G = zeros(l);
for i = 1:l
    for j = i+1:l
        if norm(X_true(i,:) - X_true(j,:)) <= theta
            G(i,j) = 1;
            G(j,i) = 1;
        end
    end
end

% Generate noisy pairwise distances
D_noisy = D_true + sigma * randn(1, length(D_true));

% Restrict the pairwise distances to the contact map graph
D_noisy(G == 0) = NaN;

% Recover the 3D structure using multidimensional scaling
D_recovered = squareform(D_noisy);
X_recovered = mdscale(D_recovered, 3);

% Visualize the results
figure;
scatter3(X_true(:,1), X_true(:,2), X_true(:,3), 'filled');
hold on;
scatter3(X_recovered(:,1), X_recovered(:,2), X_recovered(:,3), 'filled');
legend('True structure', 'Recovered structure');