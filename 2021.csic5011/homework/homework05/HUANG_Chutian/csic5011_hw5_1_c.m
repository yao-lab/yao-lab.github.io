clear
clc
m = 20;
n = 20;
lambda = 0.25; %regularization parameter
p = 0.045;
A = randn(m,n);
[U, S, V] = svd(A); %A=U*S*V', S in descending order
E0 = rand(m,n);
E = E0.*abs(E0<p);

figure

for r=1:8 %rank of matrix L
L = U(:,1:r) * V(:,1:r)';
M = L + E;

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

%fprintf('inf norm of S-E is %e \n',norm(S-E,'inf'))
%fprintf('2 norm of L-R is %e',norm(L-R))

plot(r,norm(S-E,'inf'),'ro',r,norm(L-R),'g+')
hold on
%plot(p,norm(L-R),'g+')
%hold on
end
hold off
xlabel('rank r')
ylabel('||S-E||_{\infty},||L-R||_2')
legend('||S-E||_{\infty}','||L-R||_2')