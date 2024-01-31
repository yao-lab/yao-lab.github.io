m=20; n=20; p=0.9;
prob = zeros(m, 1);
R = randn(m, n);
lambda = 0.25;
total_iterations = 20;
for r =1:20
    [U, S, V] = svds(R, r);
    L0 = U(:, 1:r)*V(:, 1:r)';
    E = rand(m);
    % t is the totla success number under current p
    t = 0;
    for iter = 1:total_iterations
        [U, S, V] = svds(R, r);
        L0 = U(:, 1:r)*V(:, 1:r)';
        E = rand(m);
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
    prob(r) = t/50;
 end

disp('The probability for different r is:')
prob