function simu_data = simu_unit_cv(n, p, D, A, c, k, nu_s, option)
% the simulation unit for simulation experiments with cross validation.

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
%   simu_data.fdr: fdr of cv optimal nu in split knockoffs
%   simu_data.sd_fdr: standard error of fdr of cv optimal nu in split knockoffs
%   simu_data.power: power of cv optimal nu in split knockoffs
%   simu_data.sd_power: standard error of power of cv optimal nu in split knockoffs
%   simu_data.fdr_knock: fdr of knockoffs
%   simu_data.sd_fdr_knock: standard error of fdr of knockoffs
%   simu_data.power_knock: power of knockoffs
%   simu_data.sd_power_knock: standard error of power of knockoffs
%   simu_data.cv_list: length(nu_s) * tests matrix with cv loss for each nu
%       in each test
%   simu_data.chosen_nu: cv selected nu for each test

sigma = 1; % noise level
tests = 20; % number of experiments
num_nu = length(nu_s);

% generate X
Sigma = zeros(p, p);
for i = 1: p
    for j = 1: p
        Sigma(i, j) = c^(abs(i - j));
    end
end

m = size(D, 1);

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
fdr_cv = zeros(tests, 1);
power_cv = zeros(tests, 1);
cv_list = zeros(num_nu, tests);
chosen_nu = zeros(tests, 1);
fdr_knockoff = zeros(tests, 1);
power_knockoff = zeros(tests, 1);

%%%%%%%%%%%%%%% begin simulation %%%%%%%%%%%%%

for test = 1: tests

    % generate varepsilon
    rng(test);
    
    % generate noise and y
    varepsilon = randn(n, 1) * sqrt(sigma);
    y = X * beta_true + varepsilon;
   
    
    % running knockoff as a comparison
    if m <= p
        result = split_knockoffs.private.convert_knockoff(X, D, y, option);
        [fdr_knockoff(test, 1), power_knockoff(test, 1)] = split_knockoffs.private.simu_eval(gamma_true, result);
    end
    
    [result, cv_list(:, test), chosen_nu(test)] = split_knockoffs.cv_filter(X, D, y, nu_s, option);
    [fdr_cv(test, 1), power_cv(test, 1)] = split_knockoffs.private.simu_eval(gamma_true, result);
end

% compute the means

simu_data = struct;

fdr = mean(fdr_cv);
power = mean(power_cv);
sd_fdr= sqrt(var(fdr_cv));
sd_power = sqrt(var(power_cv));

mean_fdr_knockoff = mean(fdr_knockoff);
mean_power_knockoff = mean(power_knockoff);
sd_fdr_knockoff= sqrt(var(fdr_knockoff));
sd_power_knockoff = sqrt(var(power_knockoff));

simu_data.fdr = fdr;
simu_data.power = power;
simu_data.sd_fdr = sd_fdr;
simu_data.sd_power = sd_power;

simu_data.fdr_knock = mean_fdr_knockoff;
simu_data.power_knock = mean_power_knockoff;
simu_data.sd_fdr_knock = sd_fdr_knockoff;
simu_data.sd_power_knock = sd_power_knockoff;

simu_data.cv_loss = cv_list;
simu_data.chosen_nu = chosen_nu;

end