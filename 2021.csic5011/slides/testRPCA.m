% Construct a random 20-by-20 Gaussian matrix and construct a rank-1
% matrix using its top-1 singular vectors
R = randn(20,20);
[U,S,V] = svds(R,3);
L0 = U(:,1)*V(:,1)';

% Construct a 90% uniformly sparse matrix
E0 = rand(20);
S0 = 1*abs(E0>0.9);
X = L0 + S0;

% Choose the regularization parameter
lambda = 0.25;

% Solve the SDP by calling cvx toolbox
if exist('cvx_setup.m','file'),
    cvx_setup
end

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

% Another simple CVX implementation directly using matrix nuclear norm
    
cvx_begin
    cvx_precision low
    variables X_1(20,20) X_2(20,20)  
    minimize(norm_nuc(X_1)+lambda*norm(X_2(:),1))
    subject to
        X_1 + X_2 == X;
cvx_end

% The difference between sparse solution X_2 and true S0
disp('$\|X_2-S0\|_infty$:')
norm(X_2-S0,'inf')
% The difference between the low rank solution X_1 and true L0
disp('$\|X_1-L0\|$:')
norm(X_1-L0)