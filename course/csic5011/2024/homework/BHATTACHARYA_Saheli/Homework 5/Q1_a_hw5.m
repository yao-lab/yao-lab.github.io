% CSIC:5011, HW-5, Q1(a)
clear; clc;

m=20; n=20;
r=4; %r=2 leads to failure
A=randn(m,n);
[U,S,V] = svds(A,r);

L0 = U(:,1:r)*V(:,1:r)'; %Low rank-r matrix

% Construct a 90% uniformly sparse matrix
p=0.05;
E0 = rand(20);
S0 = 1*abs(E0>(1-p));
X = L0 + S0;

lambda = 0.25; %regularization parameter

cvx_begin
    variable L(m,n);
    variable S(m,n);
    variable W1(m,n);
    variable W2(m,n);
    variable Y(2*m,2*n) symmetric;
    Y == semidefinite(2*m);
    minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S))));
    subject to
        L + S >= X-1e-5; % L+S == X
        L + S <= X + 1e-5;
        Y == [W1, L;L' W2];
cvx_end

% The difference between sparse solution S and true S0
disp('$\|S-S0\|_infty$:')
norm(S-S0,'inf')
% The difference between the low rank solution L and true L0
disp('$\|L-L0\|$:')
norm(L-L0)

% The difference between sparse solution S and true S0
disp('$\|S-S0\|_fro$:')
norm(S-S0,'fro')
% The difference between the low rank solution L and true L0
disp('$\|L-L0\|_fro$:')
norm(L-L0,'fro')

%Part(b)
if (norm(S-S0,'fro')<1e-3 && norm(L-L0,'fro')<1e-3)
    disp('Success') %p=0.1 to 0.2 leads to success 
else
    disp('Failure') %p=0.3 leads to failure
end

