% CSIC:5011, HW-5, Q1(a)
clear; clc;

m=20; n=20;
r=1; %r=2 leads to failure
A=randn(m,n);
[U,S,V] = svds(A,r);

L0 = U(:,1:r)*V(:,1:r)'; %Low rank-r matrix

% Construct a 90% uniformly sparse matrix
p=[0.05 0.1 0.15 0.2 0.25 0.3];

succ_prob=zeros(1,length(p));
Niter=20;

for i=1:length(p)
    succ=0;
    for j=1:Niter
        E0 = rand(m,n);
        S0 = 1*abs(E0>(1-p(i)));
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


%Part(b)
if (norm(S-S0,'fro')<1e-3 && norm(L-L0,'fro')<1e-3)
    succ=succ+1;
end
    end
    
succ_prob(i)=succ/Niter;
end

figure(1)
plot(p,succ_prob);
xlabel('p')
ylabel('Success Probability')
title('Success Prob. v/s sparsity level (p), r=1')

