
function [success] = problem1d(p,r)
    m=20;
    n=20;
    R = randn(m,n);
    [U,S,V] = svds(R,10);
    
    A = U(:,1:r)*V(:,1:r)';

    E0 = rand(m);
    E = 1*abs( E0>(1-p));
    M = A + E;

    lambda = 0.25;
    
    S=0;
    Y=0;
    mu=0.25*m*n/norm(M,1);
    
    
    for i=1:100
        L=M-S-1/mu*Y;
        [L_u,L_s,L_v] = svd(L,'econ');
        L_s=wthresh(L_s,'s',mu);
        L=L_u*L_s*L_v';
        
        S=wthresh(M-L+(1/mu)*Y,'s',lambda*mu); 
        Y=Y+mu*(M-L-S);
        norm(S-E,'inf')
    end
    
    norm(S-E,'inf')
    norm(A-L);
    success=norm(S-E,'inf')<1e-4;
end

