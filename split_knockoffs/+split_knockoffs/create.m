function [A_beta,A_gamma,tilde_y,tilde_A_gamma] = create(X, y, D, nu, option)
% split_knockoffs.create gives the variable splitting design matrix
% [A_beta, A_gamma] and response vector tilde_y. It will also create a
% knockoff copy for A_gamma if required.

% Input Argument:
% X : the design matrix
% y : the response vector
% D : the linear transform
% nu: the parameter for variable splitting
% option: options for creating the Knockoff copy.
% % option.copy = true : create a knockoff copy
% % option.eta : the choice of eta for creating the knockoff copy

% Output Argument:
% A_beta: the design matrix for beta after variable splitting
% A_gamma: the design matrix for gamma after variable splitting
% tilde_y: the response vector after variable splitting
% tilde_A_gamma: the knockoff copy of A_beta; will be [] if option.copy =
% false.


[n,~] = size(X);
m = size(D,1);

% calculate A_beta, A_gamma
A_beta = [X/sqrt(n);D/sqrt(nu)];
A_gamma = [zeros(n,m);-eye(m)/sqrt(nu)];

% calculate tilde_y
tilde_y = [y/sqrt(n);zeros(m,1)];

if option.copy == true
    
    s_size = 2-option.eta;

    % calculte inverse for Sigma_{beta, beta}
    Sigma_bb = A_beta' * A_beta;
    if sum(abs(eig(Sigma_bb))>1e-6) == size(Sigma_bb,1)
        Sigma_bb_inv = inv(Sigma_bb);
    else
        Sigma_bb_inv = pinv(Sigma_bb);
    end

    % calculate Sigma_{gamma, gamma}, etc
    [~,S,V] = knockoffs.private.canonicalSVD(A_gamma);
    Sigma_gg = V * sparse(S.^2) * V';
    Sigma_gb = A_gamma' * A_beta;
    Sigma_bg = Sigma_gb';

    % calculate C_nu
    C = Sigma_gg - Sigma_gb * Sigma_bb_inv * Sigma_bg;
    C = (C + C')/2;
    C_inv = inv(C);

    % generate s
    diag_s = sparse(diag(min(s_size * min(eig(C)), 1/nu) * ones(size(C,1),1)));


    % calculate K^T K = 2S-S C_nu^{-1} S
    KK = 2 * diag_s - diag_s * C_inv * diag_s;
    KK = (KK +KK)/2;
    [Uee,See] = eig(KK);
    K = Uee * sqrt(See) * Uee';

    % calculate U=[U1;U_2] where U_2 = 0_m* m
    %U_1 is an orthogonal complement of X
    [~,~,~,U_perp] = knockoffs.private.decompose(X, []);
    U_1 = U_perp(:,1:m);
    U = [U_1; zeros(m)];

    % calculate sigma_beta beta^{-1} sigma_beta gamma
    short = Sigma_bb_inv * Sigma_bg;


    % calculate tilde_A_gamma
    tilde_A_gamma = A_gamma * (eye(m) - C_inv * diag_s) + A_beta * short * C_inv * diag_s + U * K;

else
    tilde_A_gamma = [];
end
end