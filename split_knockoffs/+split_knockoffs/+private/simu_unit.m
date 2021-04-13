function simu_data = simu_unit(n, p, D, A, c, k , nu_s, option)
% the simulation unit for simulation experiments.

% input argument
% n: the sample size
% p: the dimension of variables
% D: the linear transform
% A: SNR
% c: feature correlation
% k: number of nonnulls in beta
% nu_s: a set of nu to choose from
% option: option for split knockoffs

% output argument
% simu_data: a structure contains the following elements
%   simu_data.fdr_split: a vector recording fdr of split knockoffs w.r.t.
%       nu
%   simu_data.power_split: a vector recording power of split knockoffs w.r.t.
%       nu
%   simu_data.fdr_knock: fdr of knockoffs
%   simu_data.power_knock: power of knockoffs

sigma = 1; % noise level
tests = 20; % number of experiments
num_nu = length(nu_s);
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

% create matrices to store results
fdr_split = zeros(tests, num_nu);
power_split = zeros(tests, num_nu);
fdr_knockoff = zeros(tests, 1);
power_knockoff = zeros(tests, 1);

%%%%%%%%%%%%%%% begin simulation %%%%%%%%%%%%%

for test = 1: tests

    % generate varepsilon
    rng(test);
    
    % generate noise and y
    varepsilon = randn(n, 1) * sqrt(sigma);
    y = X * beta_true + varepsilon;
    
    if m <= p
        result = split_knockoffs.private.convert_knockoff(X, D, y, option);
        [fdr_knockoff(test, 1), power_knockoff(test, 1)] = split_knockoffs.private.simu_eval(gamma_true, result);
    end
    
    results = split_knockoffs.filter(X, D, y, nu_s, option);

    for i = 1: num_nu
        result = results{i};
        [fdr_split(test, i), power_split(test, i)] = split_knockoffs.private.simu_eval(gamma_true, result);
    end
    
end

% compute the means

mean_fdr_split = mean(fdr_split);
mean_power_split = mean(power_split);

mean_fdr_knockoff = mean(fdr_knockoff);
mean_power_knockoff = mean(power_knockoff);

simu_data = struct;

simu_data.fdr_split = mean_fdr_split;
simu_data.power_split = mean_power_split;

simu_data.fdr_knock = mean_fdr_knockoff;
simu_data.power_knock = mean_power_knockoff;

end