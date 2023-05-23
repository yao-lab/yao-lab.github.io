function [c,J_plus,T,Acc] = TPT2(A,c0,source,react)
%TPT2 Summary of this function goes here
%  the source/react states are given  

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
q(source) = 0;
q(react) = 1;
S = setdiff([1:n],[source,react]);
L = eye(n)-P;
q(S) = -L(S,S)^-1*L(S,react);

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
T(source) = sum(J_plus(source,:),2);
T(S) = sum(J_plus(S,:),2);
T(react) = sum(J_plus(:,react));
end


