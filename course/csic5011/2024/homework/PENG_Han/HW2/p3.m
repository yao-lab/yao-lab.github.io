clear
clc

n = 800;
W = normrnd(0,sqrt(1/(4*n)),n,n);
W1 = triu(W);
W2 = W1;
W2 = W2 - diag(diag(W2));
W2 = W2';
Wi = W1 + W2;
[~,D] = eig(Wi);
lam = diag(D);
histogram(lam,20)


