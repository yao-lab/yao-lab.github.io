function [A, labels] = DCSBM(n, B, theta)
% Generate a random graph from a degree-corrected stochastic block model
% with n nodes, B blocks, and degree exponent theta.

% Generate the block membership vector
labels = repelem(1:B, floor(n/B));
labels = [labels, randperm(B, n - length(labels))];

% Generate the degree sequence
deg = zeros(n, 1);
for i = 1:B
    idx = find(labels == i);
    deg(idx) = round(power(rand(length(idx), 1), 1/(theta-1)));
end

% Generate the adjacency matrix
A = zeros(n, n);
for i = 1:n
    for j = i+1:n
        if labels(i) == labels(j)
            A(i,j) = A(j,i) = rand() < (deg(i)*deg(j)) / sum(deg(labels==labels(i)));
        else
            A(i,j) = A(j,i) = rand() < (deg(i)*deg(j)) / sum(deg(labels==labels(i))) / sum(deg(labels==labels(j)));
        end
    end
end