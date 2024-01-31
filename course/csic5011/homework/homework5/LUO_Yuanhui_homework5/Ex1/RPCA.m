m=20; n=20; r = 1; p=0.1;
R = randn(m, n);
[U, S, V] = svds(R, r);
L0 = U(:, 1:r)*V(:, 1:r)';
E = rand(m);
S0 = 1* abs(E>p);
M0 = L0 + S0;
lambda = 0.25;

if exist('cvx_setup.m', 'file')
    cvx_setup
end

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

% The difference between sparse solution S and the true S0
disp('$\|S-S0\|_INFTY$:')
norm(S-S0, 'inf')
% The difffereence between the low rank solution L and true L0
disp('$\|L-L0\|$:$')
norm(L-L0)
% Whether the revovery is success
disp('The successful recovery is:')
(norm(L-L0)/norm(L0)<0.01)



