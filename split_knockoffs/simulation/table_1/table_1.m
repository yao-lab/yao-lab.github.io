% This file reproduces table 1 for split Knockoffs. That is the performance
% of split knockoffs with cross validation. The result for split Knockoffs
% with cv can be found at fdr_split, power_split, the result for knockoffs
% can be found at fdr_knock, power_knock. The cv loss for each experiment
% and each nu can be found at cv_list. Each cell in cv_list is a
% length(nu_s) * tests matrix with cv loss for each nu.

root = pwd;
base_root = erase(pwd, "\simulation\table_1");
addpath(base_root);

k = 20; % sparsity level
A = 1;% magnitude
n = 350;% sample size
p = 100;% dimension of variables
c = 0.5; % feature correlation

option.q = 0.2;
option.eta = 0.1;
option.stage0 = "path";
option.normalize = true;
option.cv_rule = "min";
option.lambda_s = 10.^[0: -0.01: -6];
option.k_fold = 7;

% settings for nu
nu_s = 10.^[-1: 0.4: 3];
num_nu = length(nu_s);

fdr_split = zeros(2, 3);
power_split = zeros(2, 3);
fdr_knock = zeros(2, 2);
power_knock = zeros(2, 2);

sd_fdr_split = zeros(2, 3);
sd_power_split = zeros(2, 3);
sd_fdr_knock = zeros(2, 2);
sd_power_knock = zeros(2, 2);

cv_list = cell(2, 3);
chosen_nu = cell(2, 3);

num_method = 2;
method_s = ["knockoff", "knockoff+"];

% generate D
D_G = zeros(p-1, p);

for i = 1:(p-1)
    D_G(i, i) = 1;
    D_G(i, i+1) = -1;
end

D_1 = eye(p);
D_2 = D_G;
D_3 = [eye(p); D_G];
D_s = {D_1, D_2, D_3};

for meth = 1: 1
    method = method_s(meth);
    option.method = method;
    for D_choice = 1: 3
        D = D_s{D_choice};
        simu_data = split_knockoffs.private.simu_unit_cv(n, p, D, A, c, k, nu_s, option);
        fdr_split(meth, D_choice) = simu_data.fdr;
        power_split(meth, D_choice) = simu_data.power;
        sd_fdr_split(meth, D_choice) = simu_data.sd_fdr;
        sd_power_split(meth, D_choice) = simu_data.sd_power;
        if D_choice < 3
            fdr_knock(meth, D_choice) = simu_data.fdr_knock;
            power_knock(meth, D_choice) = simu_data.power_knock;
            sd_fdr_knock(meth, D_choice) = simu_data.sd_fdr_knock;
            sd_power_knock(meth, D_choice) = simu_data.sd_power_knock;
        end
        cv_list{meth, D_choice} = simu_data.cv_loss;
        chosen_nu{meth, D_choice} = simu_data.chosen_nu;
    end
end

%% plot for cross validation loss

for i = 1: 3
    mean_cv = mean(cv_list{1, i}, 2);
    x = [-1:0.4:3];
    fig = figure();
    hold on
    grid on
    set(fig, 'DefaultTextInterpreter', 'latex');
    plot(x, mean_cv);
    hold off

    set(gca,'XTick',[-1:0.4:3]);
    xlabel('$\log_{10} (\nu)$');
    ylabel('Cross Validation Loss');
end

