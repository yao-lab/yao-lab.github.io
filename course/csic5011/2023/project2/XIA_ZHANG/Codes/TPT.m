function [c,J_plus,T,Acc] = TPT(A,c0)
%TPT Summary of this function goes here
%   Detailed explanation goes here

% Markov Chain 
d = sum(A,2);
D = diag(d);
P = D^(-1)*A;

[V,Lam] = eigs(P');
idx = find(abs(diag(Lam)-1)<1e-8,1);
pi = V(:,idx)';
pi = pi/sum(pi);

% the committor function
n = length(pi);
q = zeros(n,1);
q(1) = 0;
q(n) = 1;
S = 2:1:n-1;
L = eye(n)-P;
q(S) = -L(S,S)^-1*L(S,n);

% Graph decomposition
c = zeros(n,1);
c(q<0.5)=0;
c(q>=0.5)=1;

% Acc
Acc = length(find(c==c0))/n;


% Effective flux
J = zeros(n,n);
for x = 1:n
    for y = 1:n
        J(x,y) = pi(x)*(1-q(x))*P(x,y)*q(y);
    end
end

J_plus = max(J-J',0);

% Transition flux
T = zeros(n,1);
T(1:n-1) = sum(J_plus(1:n-1,:),2);
T(n) = sum(J_plus(:,n));
end

