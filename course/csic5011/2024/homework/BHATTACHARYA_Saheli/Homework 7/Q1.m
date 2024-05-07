%Q1(a)
clear; clc;
load univ_cn.mat W_cn univ_cn rank_cn

v = rank_cn;        % research rank of universities
webpage = univ_cn;  % webpage of universities in mainland china
W = W_cn;           % Link weight matrix

D = sum(W,2);
n = length(D);
idnz = find(D>0);
T = zeros(n,n);
T(idnz,idnz) = diag(1./D(idnz)) * W(idnz,idnz);
alpha=0.85;

T1 = alpha * T + (1-alpha)*ones(n,1)*ones(1,n)/n;
[evec,eval] = eigs(T1',1);
score_page=evec/sum(evec);  % pagerank score

% Google PageRank for \alpha=0.85
[~,idp]=sort(score_page,'descend'); %idp contains the page Google pagerank

%Showing top 10 rankings
webpage{idp(1:10)}

%Q1(b)

% HITS rank
[U,S,V] = svds(W);
u1 = U(:,1)/sum(U(:,1));
v1 = V(:,1)/sum(V(:,1));
[~,idu]=sort(u1,'descend');
[~,idv]=sort(v1,'descend');
webpage{idu(1:10)}   % Hub Ranking
webpage{idv(1:10)}   % Authority Ranking

%Q1(c)
[rho_google_pr, ~]=corr(idp,v,'type','Kendall'); %kendall tau distance for Google Pagerank

[rho_HITS_hub_pr, ~]=corr(idu,v,'type','Kendall'); %kendall tau distance for HITS hub Pagerank

[rho_HITS_authority_pr, ~]=corr(idv,v,'type','Kendall'); %kendall tau distance for HITS hub Pagerank

%Q1(d)

alpha1 = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.85 0.9]; % alpha=0.85 is Google's PageRank choice

for i=1:length(alpha1)
	T1 = alpha1(i) * T + (1-alpha1(i))*ones(n,1)*ones(1,n)/n;
    [evec1,eval1] = eigs(T1',1);
	[score_page1(:,i)]=evec1/sum(evec1);  % pagerank score
    [~,idp1]=sort(score_page1(:,i),'descend');
    [rho_kendall(i),~]=corr(idp1,v,'type','Kendall');
end

figure(1)
plot(alpha1,rho_kendall)
xlabel('\alpha')
ylabel('Kendalls \tau distance to Research Ranking')
