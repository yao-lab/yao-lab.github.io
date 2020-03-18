% Construct a random 20-by-20 Gaussian matrix and construct a rank-1
% matrix using its top-1 singular vectors
R = randn(20,20);
[U,S,V] = svds(R,3);
A = U(:,1)*V(:,1)';

% Construct a 90% uniformly sparse matrix
E0 = rand(20);
E = 1*abs(E0>0.9);
X = A + E;

% Choose the regularization parameter
lambda = 0.25;

% Solve the SDP by calling cvx toolbox
if exist('cvx_setup.m','le'),
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
        L + S >= X-1e-5;
        L + S <= X + 1e-5;
        Y == [W1, L';L W2];
cvx_end

% The difference between sparse solution S and E
disp('$\|S-En\|_infty$:')
norm(S-E,'inf')
% The difference between the low rank solution L and A
disp('$\|A-L\|$:')
norm(A-L)