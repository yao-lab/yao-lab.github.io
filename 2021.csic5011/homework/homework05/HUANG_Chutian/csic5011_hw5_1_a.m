clear
clc
m = 20;
n = 20;
r = 1; %rank of matrix L
A = randn(m,n);
[U, S, V] = svd(A); %A=U*S*V', S in descending order
L = U(:,1:r) * V(:,1:r)';
E0 = rand(m,n);
p = 0.1;
E = E0.*abs(E0<p);
M = L + E;

%cvx_setup

lambda = 0.25; %regularization parameter

cvx_begin
variable R(m,n);%recover of low rank matrix L
variable S(m,n);%recover of sparse matrix E
variable W1(m,m);
variable W2(n,n);
variable Y(m+n,m+n) symmetric;
Y == semidefinite(40);
minimize(0.5*trace(W1) + 0.5*trace(W2)+lambda*sum(sum(abs(S))));
subject to
R + S >= M-1e-5
R + S <= M+1e-5
Y == [W1 R;R' W2];
cvx_end

fprintf('inf norm of S-E is %e \n',norm(S-E,'inf'))
fprintf('2 norm of L-R is %e',norm(L-R))


