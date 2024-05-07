clc;
clear all;
close all;

m=20; n=20; r = 1;
prob = zeros(10, 1);
R = randn(m, n);
[U, S, V] = svds(R, r);
L0 = U(:, 1:r)*V(:, 1:r)';
E = rand(m);
lambda = 0.25;
total_iterations = 20
for i =1:10
    p = i/10
    % t is the totla success number under current p
    t = 0;
    for iter = 1:total_iterations
        S0 = 1* abs(E>p);
        M0 = L0 + S0;
        cvx_begin
            variable L(m,n);
            variable S(m,n);
            variable W1(m,n);
            variable W2(m,n);
            variable Y(m+n,m+n) symmetric;
            Y == semidefinite(m+n);
            minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S))));
            subject to
                L + S >= M0 - 1e-5;
                L + S <= M0 + 1e-5;
                Y == [W1, L'; L, W2];
        cvx_end
        if (norm(L-L0)/norm(L0)<0.01)
            t = t+1;
        end
    end
    prob(i) = t/50;
 end

disp('The probability for different p is:')
prob
