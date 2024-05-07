HLM = importdata('HongLouMeng374.txt');
X = HLM.data;
A = X*X';
D = diag(sum(A));
L = D - A;
[V, lambda] = eig(L);
lambda = diag(lambda);
[lambda, idx] = sort(lambda,'ascend');
lambda2 = lambda(2);
v2 = V(:, idx(2));
[f, order] = sort(v2,'ascend');

% Construct the subsets Si
n = length(f);
Si = cell(n, 1);
for i = 1:n
    Si{i} = order(1:i);
end

% Find the optimal subset S*
min_value = Inf;
S_star = [];
for i = 1:n-1
    cut = cut_size(Si{i}, Si{i+1}, A);
    value = (cut / sum(sum(A(Si{i},:)))) + (cut / sum(sum(A(Si{i+1},:))));
    if value < min_value
        min_value = value;
        S_star = Si{i};
    end
end

% (d) lambda_2 is not larger than alpha_f

S_plus = find(v2 >= 0);
S_minus = find(v2 < 0);

% Compute the Cheeger ratio
cut = sum(sum(A(S_plus, S_minus)));
vol_plus = sum(sum(A(S_plus, :)));
vol_minus = sum(sum(A(S_minus, :)));
h = cut / min(vol_plus, vol_minus);


function cut = cut_size(S1, S2, A)
    cut = sum(sum(A(S1, S2)));
end