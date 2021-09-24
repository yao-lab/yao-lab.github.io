
function [success] = problem1(p,r)
    R = randn(20,20);
    [U,S,V] = svds(R,10);
    
    A = U(:,1:r)*V(:,1:r)';

    E0 = rand(20);
    E = 1*abs(E0>(1-p));
    X = A + E;

    lambda = 0.25;
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
    L + S <= X + 1e-5; Y == [W1, L';L W2];
    cvx_end
    % The difference between sparse solution S and E
    disp('$\—S-E\— \infty$:') ;
    norm(S-E,'inf');
    % The difference between the low rank solution L and A
    disp('\—A-L\—') ;
    norm(A-L);
    success=norm(S-E,'inf')<1e-4;
end

