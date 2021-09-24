m = 20;
n = 20;
r = 1;
p = 0.9;

A = randn(m,n);
[u,s,v] = svd(A);
s2 = zeros(m,n);
for i=1:r
    s2(i,i) = s(i,i);
end

L0 = u*s2*v';
lambda = 0.25;
E0 = rand(20);

E = 1*abs(E0>p);

M = L0+E;



cvx_begin
variable L(20,20); 
variable S(20,20); 
variable W1(20,20); 
variable W2(20,20); 
variable Y(40,40) symmetric; 
Y == semidefinite(40); 
minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S)))); 
subject to
    L + S >= M - 1e-10;
    L + S <= M + 1e-10;
    Y == [W1, L';L W2];
cvx_end

err = sum((S-E)^2,[1,2]);
err2 = sum((L0-L)^2,[1,2]);

