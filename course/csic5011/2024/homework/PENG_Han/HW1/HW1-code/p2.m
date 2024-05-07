clear
clc


% My MDS - Bay area
if 1
n = 7;
D = load('D1.txt');
D = D.*D;
H = eye(n) - 1/n*ones(n,n);
K = -1/2*H*D*H';
[~,S,V] = svd(K);

%plot lambda
lam=zeros(1,n);
for i=1:n
    lam(i) = S(i,i);
end
figure(1)
plot(1:n,lam/trace(S));
xlabel('i')
ylabel('lam/trace(S)')
title('Bay area cities')

% plot coordinates
figure(2)
S_sqrt = sqrt(S);
Y = S_sqrt*V';
k = 2;
Y = Y(1:k,:);
plot(Y(1,:),Y(2,:),'.','MarkerSize',20)
labels = {'berkeley','los angeles','san diego','sacramento','napa','san jose', 'livermore'};
for i = 1:n
    text(Y(1,i), Y(2,i), labels{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right','FontSize', 13)
end
title('My MDS')
end




% MATLAB Version  - Bay area
if 1
D = load('D1.txt');
%D = D.*D;
distance_matrix = D;
% Perform MDS on the distance matrix
city_coordinates = mdscale(distance_matrix, 2);
%city_coordinates = cmdscale(distance_matrix, 2);
Y = city_coordinates;

% Print the computed city coordinates
figure(3)
plot(Y(:,1),Y(:,2),'.','MarkerSize',20)
labels = {'berkeley','los angeles','san diego','sacramento','napa','san jose', 'livermore'};
for i = 1:numel(Y(:,1))
    text(Y(i,1), Y(i,2), labels{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right','FontSize', 13)
end
title('MATLAB FUNCTION')
end





% My MDS - China
if 1
n = 8;
D = load('D.txt');
D = D.*D;
H = eye(n) - 1/n*ones(n,n);
K = -1/2*H*D*H';
[~,S,V] = svd(K);

%plot lambda
lam=zeros(1,n);
for i=1:n
    lam(i) = S(i,i);
end
figure(4)
plot(1:n,lam/trace(S));
xlabel('i')
ylabel('lam/trace(S)')
title('China cities')

% plot coordinates
figure(5)
S_sqrt = sqrt(S);
Y = S_sqrt*V';
k = 2;
Y = Y(1:k,:);
plot(Y(1,:),Y(2,:),'.','MarkerSize',20)
labels = {'Beijing','Shanghai','Hong Kong','Changsha','Xi An', 'Kunming','Harbin','Wulumuqi'};
for i = 1:n
    text(Y(1,i), Y(2,i), labels{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right','FontSize', 13)
end
title('My MDS - China')
end






% chatgpt1  - Bay area
if 0
% Load dissimilarity matrix or distance matrix
D = load('D1.txt');
D = D.*D;

% Number of points
n = size(D, 1);

% Centering matrix
H = eye(n) - (1/n) * ones(n, n);

% Double centering
B = -0.5 * H * D * H';

% Perform eigenvalue decomposition
[U, S, ~] = svd(B);

% Number of dimensions to keep
k = 2;

% Compute the MDS coordinates
Y = U(:, 1:k) * sqrt(S(1:k, 1:k));

% Plot the MDS coordinates
scatter(Y(:, 1), Y(:, 2), 'filled');
labels = {'berkeley','los angeles','san diego','sacramento','napa','san jose', 'livermore'}; % Add your own labels
text(Y(:, 1), Y(:, 2), labels, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

% Set axis labels and title
xlabel('Dimension 1');
ylabel('Dimension 2');
title('Multidimensional Scaling');

% Optional: Adjust the aspect ratio of the plot
%axis equal;

end



% chatgpt2  - Bay area
if 0
temp = load('D1.txt');
temp = temp.*temp;
% Sample similarity matrix
similarity_matrix = temp;

% Number of dimensions for the reduced space
num_dimensions = 2;

% Number of objects
num_objects = size(similarity_matrix, 1);

% Centering matrix
J = eye(num_objects) - (1/num_objects) * ones(num_objects);

% Double centering
B = -0.5 * J * similarity_matrix * J;

% Eigenvalue decomposition
[V, D] = eig(B);

% Sort eigenvalues and eigenvectors in descending order
[eigenvalues_sorted, indices] = sort(diag(D), 'descend');
eigenvectors_sorted = V(:, indices);

% Select the desired number of dimensions
selected_eigenvalues = eigenvalues_sorted(1:num_dimensions);
selected_eigenvectors = eigenvectors_sorted(:, 1:num_dimensions);

% Compute the coordinates of the objects in the reduced space
Y = selected_eigenvectors * sqrt(diag(selected_eigenvalues));

% Plot the results
scatter(-Y(:,1), -Y(:,2));
end

