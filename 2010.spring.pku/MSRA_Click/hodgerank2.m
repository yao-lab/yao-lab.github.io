% The second model.

numNode = 89; %length(uid);

ID=1:numNode;

T1=TT_click(ID,ID);
N1=NN_click(ID,ID);
G = ((T1+T1')>0);
G = G - diag(diag(G));

% Find the maximal connect component in G.
if ~exist('matlab_bgl','dir'),
    addpath ../../matlab_tools/matlab_bgl
end
[ci,csize]=components(sparse(G));
ind = find(ci==1);
G = G(ind,ind);
T1 = T1(ind,ind);
N2 = N1(ind,ind);

% Find Edge set of MCC(G).
[i,j]=find(triu(G,1)>0);
numEdge = length(i);

% Typo here: replace d0 by d0' in the following for consistency with
% literature. 
d0 = zeros(length(ind),numEdge);
w = zeros(numEdge,1);
N = w;
for e=1:numEdge,
    d0(i(e),e) = 1;
    d0(j(e),e) = -1;
    w(e) = T1(i(e),j(e))-T1(j(e),i(e));
    N(e) = N2(i(e),j(e));
end

% Triangle FOrmation.
numTriangle = 0;
E = [i,j];
T = [];
for k=1:numEdge,
    tmp = [i(k) j(k)];  % e.g. [1 2]
    t = find(j(k)==E(:,1)); % [2,*]
    for l=1:length(t),      
        s = setdiff(find(E(t(l),2)==E(:,2)),t(l));    % [?,*]
        for m=1:length(s),
            if E(s(m),1)==i(k),
                numTriangle = numTriangle + 1;
                T = [T; tmp E(s(m),2)];
            end
        end
    end
end

d1 = zeros(numEdge,numTriangle);
for k=1:numTriangle,
    e1 = find(E(:,1)==T(k,1) & E(:,2)==T(k,2));
    e2 = find(E(:,1)==T(k,2) & E(:,2)==T(k,3));
    e3 = find(E(:,1)==T(k,1) & E(:,2)==T(k,3));
    d1(e2,k)=1;
    d1(e3,k)=-1;
end

% Solve the least square x_hodge = argmin_x norm(d0'*x - w)
L0 = d0*diag(N)*d0';
rank(L0)
ndiv = d0*w;
% Least square solution.
x_hodge = L0\ndiv; %lsqr(L0,ndiv);


L1 = d0'*d0 + d1*d1';

rank(L1)
%[v,d]=eigs(100-L1);

plot(ID(ind),p_click(ID(ind))/sum(p_click(ID(ind))),'b-.x',ID(ind),(x_hodge-min(x_hodge))/sum(x_hodge-min(x_hodge)),'r-o')
xlabel('url ID')
ylabel('Percentage')
title('Pairwise Comparison vs. Frequency of Clicks')