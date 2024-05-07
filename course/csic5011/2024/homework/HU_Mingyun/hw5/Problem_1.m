% (a)
m = 20;
n = 20;
r = 1;
p = 0.1;
R = randn(m,n);
[U,S,V] = svds(R,20);
L0 = zeros(m,n);
for i = 1:r
    L0 = L0 + U(:,i)*V(:,i)';
end
E0 = rand(20);
S0 = 1*abs(E0>(1-p));
X = L0 + E0;

lambda = 0.25;
cvx_begin
    variable L(20,20);
    variable S(20,20);
    variable W1(20,20);
    variable W2(20,20);
    variable Y(40,40) symmetric;
    Y == semidefinite(40);
    minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S))));
    subject to
        L + S >= X-1e-5; % L+S == X
        L + S <= X + 1e-5;
        Y == [W1, L';L W2];
cvx_end

% The difference between sparse solution S and true S0
disp('$\|S-S0\|_infty$:')
norm(S-S0,'inf')
% The difference between the low rank solution L and true L0
disp('$\|L-L0\|$:')
norm(L-L0)

% (b)
p_list = [0.3 0.5 0.7 0.9];
for k = 1:4
    p = p_list(k);
    R = randn(m,n);
    [U,S,V] = svds(R,20);
    L0 = zeros(m,n);
    for i = 1:r
        L0 = L0 + U(:,i)*V(:,i)';
    end
    E0 = rand(20);
    S0 = 1*abs(E0>(1-p));
    X = L0 + E0;

    lambda = 0.25;
    cvx_begin
        variable L(20,20);
        variable S(20,20);
        variable W1(20,20);
        variable W2(20,20);
        variable Y(40,40) symmetric;
        Y == semidefinite(40);
        minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S))));
        subject to
            L + S >= X-1e-5; % L+S == X
            L + S <= X + 1e-5;
            Y == [W1, L';L W2];
    cvx_end

    % The difference between sparse solution S and true S0
    disp('$\|S-S0\|_infty$:')
    norm(S-S0,'inf')
    % The difference between the low rank solution L and true L0
    disp('$\|L-L0\|$:')
    norm(L-L0)
end

% (c)
r_list = [5 10 15 20];
for k = 1:4
    r = r_list(k);
    p = 0.1;
    R = randn(m,n);
    [U,S,V] = svds(R,20);
    L0 = zeros(m,n);
    for i = 1:r
        L0 = L0 + U(:,i)*V(:,i)';
    end
    E0 = rand(20);
    S0 = 1*abs(E0>(1-p));
    X = L0 + E0;

    lambda = 0.25;
    cvx_begin
        variable L(20,20);
        variable S(20,20);
        variable W1(20,20);
        variable W2(20,20);
        variable Y(40,40) symmetric;
        Y == semidefinite(40);
        minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S))));
        subject to
            L + S >= X-1e-5; % L+S == X
            L + S <= X + 1e-5;
            Y == [W1, L';L W2];
    cvx_end

    % The difference between sparse solution S and true S0
    disp('$\|S-S0\|_infty$:')
    norm(S-S0,'inf')
    % The difference between the low rank solution L and true L0
    disp('$\|L-L0\|$:')
    norm(L-L0)
end

% (d)
m = 1000;
n = 1000;
r = 1;
p = 0.1;
R = randn(m,n);
[U,S,V] = svds(R,20);
L0 = zeros(m,n);
for i = 1:r
    L0 = L0 + U(:,i)*V(:,i)';
end
E0 = rand(1000);
S0 = 1*abs(E0>(1-p));
X = L0 + E0;

lambda = 0.25;
cvx_begin
    variable L(1000,1000);
    variable S(1000,1000);
    variable W1(1000,1000);
    variable W2(1000,1000);
    variable Y(2000,2000) symmetric;
    Y == semidefinite(2000);
    minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S))));
    subject to
        L + S >= X-1e-5; % L+S == X
        L + S <= X + 1e-5;
        Y == [W1, L';L W2];
cvx_end