% The first model.
IDs= 1:89;
T1=T_click(IDs,IDs);

% Below is the original PageRank Algorithm, but not used here.
%

% IDs = 1:length(uid);
% D1 = sum(T1,1);
% ind_nz=find(D1>0);
% ind_z = find(D1==0);
% D1_inv = ones(size(D1));
% D1_inv(ind_nz)=1./D1(ind_nz);
% DD1 = diag(D1_inv);
% I = zeros(28,1);
% I(ind_z)=1;
% P1 = T1*DD1 + diag(I);
% P2=0.85*P1+.15*ones(size(P1))/length(P1);

% A Modified Pagerank Algorithm.
T2 = 0.95*T1 + 0.05*ones(size(T1)); 
D2=diag(1./sum(T2,1));
P2=T2*D2;
[v,d]=eigs(P2);

plot(IDs,p_click(IDs)/sum(p_click(IDs)),'b-.x',IDs,v(:,1)/sum(v(:,1)),'r-o')
xlabel('url ID')
ylabel('Percentage')
title('Pairwise Comparison vs. Frequency of Clicks')