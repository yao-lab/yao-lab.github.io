% Construct a 10-by-20 Gaussian random matrix and form a 20-by-20 correlation
% (inner product) matrix R
X0 = randn(10,20);
R = X0'*X0;
d = 20;
e = ones(d,1);

% Call CVX to solve the SPCA given R
if exist('cvx_setup.m','le'),
    cvx_setup
end

% Small lambda will give dense PCA.
lambda = 0.5;
k = 10;

cvx_begin
    variable X(d,d) symmetric;
    X == semidefinite(d);
    minimize(-trace(R*X)+lambda*(e'*abs(X)*e));
    subject to
        trace(X)==1;
cvx_end

% Show the first two principal components
scatter(X(:,1),X(:,2),'o')