clear
V1 = @() randn(1)*sqrt(290);
V2 = @() randn(1)*sqrt(300);
V3 = @(a,b) -0.3*a+0.925*b+randn(1);
n = 100000;
X = zeros(n,10);
for iter = 1:n
v1 = V1();
v2 = V2();
v3 = V3(v1,v2);

x = zeros(10,1);
for i=1:4 
    x(i) = v1+randn(1);
end
for i=5:8 
    x(i) = v2+randn(1);
end
for i=9:10 
    x(i) = v3+randn(1);
end
X(iter,:) = x;
end

sigma0 = cov(X);
sigma = sigma0;
imagesc(sigma);
colorbar()

[v,d] = eig(sigma);
PCA = v(:,10:-1:7);
lambda = 1;
dim = 10;
e = ones(dim,1);

V = zeros(size(PCA));

for i =1:4
cvx_begin 
variable X(dim,dim) symmetric;
X == semidefinite(dim);
minimize(-trace(sigma*X)+lambda*(e'*abs(X)*e));
subject to 
trace(X)==1;
cvx_end
V(:,i) = X(:,1);
sigma = sigma - trace(sigma*X)*X;
end



