% This file reproduce the figure 1 on the comparison between
% split knockoffs and knockoffs

%% parameter settings
root = pwd;
base_root = erase(pwd, "\simulation\figure_1");
addpath(base_root);

n = 350;% sample size
p = 100;% dimention of variables
k = 20; % sparsity level
A = 1; % magnitude
c = 0.5; % feature correlation
sigma = 1; % noise level

option = struct;
option.q = 0.2;
option.eta = 0.1;
option.method = "knockoff";
option.stage0 = "path";
option.normalize = true;
option.lambda_s = 10.^[0: -0.01: -6];


% generate D
D_G = zeros(p-1, p);

for i = 1:(p-1)
    D_G(i, i) = 1;
    D_G(i, i+1) = -1;
end

% D = D_G;
D = eye(p);
m = size(D, 1);

% generate X
Sigma = zeros(p, p);
for i = 1: p
    for j = 1: p
        Sigma(i, j) = c^(abs(i - j));
    end
end

rng(100);
X = mvnrnd(zeros(p, 1), Sigma, n); % generate X


% generate beta and gamma
beta_true = zeros(p, 1);
for i = 1: k
    beta_true(i, 1) = A;
    if rem(i, 3) == 1
        beta_true(i, 1) = -A;
    end
end
gamma_true = D * beta_true;

S0 = find(gamma_true~=0);

% generate varepsilon
rng(1);

% generate noise and y
varepsilon = randn(n, 1) * sqrt(sigma);
y = X * beta_true + varepsilon;

%% running knockoff
if m <= p
    [~, Z] = split_knockoffs.private.convert_knockoff(X, D, y, option);

    fig = figure();
    hold on
    set(fig, 'DefaultTextInterpreter', 'latex');
    gscatter(Z(1:m), Z(m+1:2*m), ismember(1:m, S0), 'kr');
    hold off

    xlabel('Value of $Z_i$');
    ylabel('Value of $\tilde Z_i$');
    limits = [0 ceil(max(Z))];
    xlim(limits); ylim(limits);
    title('Knockoff with Lasso Statistic');
    line = refline(1,0);
    set(line, 'LineStyle', ':', 'Color', 'black');
    legend('Null feature', 'Non-null feature');
end
    

nu_s = [10];
[~, Z_path, t_Z_path] = split_knockoffs.filter(X, D, y, nu_s, option);

fig = figure();
hold on
set(fig, 'DefaultTextInterpreter', 'latex');
gscatter(Z_path{1}, t_Z_path{1}, ismember(1:m, S0), 'kr');
hold off

xlabel('Value of $Z_i$');
ylabel('Value of $\tilde Z_i$');
limits = [0 max([Z_path{1}; t_Z_path{1}])];
xlim(limits); ylim(limits);
title('Split Knockoff with Lasso Statistic');
line = refline(1,0);
set(line, 'LineStyle', ':', 'Color', 'black');
legend('Null feature', 'Non-null feature');
