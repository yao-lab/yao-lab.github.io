% load univ_cn.mat W_cn univ_cn rank_cn

v = rank_cn;        % research rank of universities
webpage = univ_cn;  % webpage of universities in mainland china
W = W_cn;           % Link weight matrix

D = sum(W,2);
n = length(D);
idnz = find(D>0);
T = zeros(n,n);
T(idnz,idnz) = diag(1./D(idnz)) * W(idnz,idnz);

% alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.85 0.9]; % alpha=0.85 is Google's PageRank choice

alpha = 0.85;

for i=1:length(alpha)
	T1 = alpha(i) * T + (1-alpha(i))*ones(n,1)*ones(1,n)/n;
    [evec,eval] = eigs(T1',1);
	[score_page(:,i)]=evec/sum(evec);  % pagerank score
end

% PageRank
[~,id]=sort(score_page(:,1),'descend');
webpage{id(1:5)}

score_out = D; % out-degree score
score_in = sum(W,1)'; % in-degree score
score_research = max(v)-v; % research score

% Authority ranking
[~,id] = sort(score_in,'descend');
webpage{id(1:5)}

% Hub ranking
[~,id] = sort(score_out,'descend');
webpage{id(1:5)}

% HITS rank
[U,S,V] = svds(W);
u1 = U(:,1)/sum(U(:,1));
v1 = V(:,1)/sum(V(:,1));
[~,idu]=sort(u1,'descend');
[~,idv]=sort(v1,'descend');
webpage{idu(1:5)}   % Hub Ranking
webpage{idv(1:5)}   % Authority Ranking

x = rank_cn;
y = id;

n = length(x); 
tau = 0;

for i = 1:n-1
    for j = i+1:n
        if (x(i) < x(j) && y(i) > y(j)) || (x(i) > x(j) && y(i) < y(j)) 
            tau = tau + 1; 
        end 
    end 
end 
tau_PageRank = tau / (n*(n-1)/2)

x = rank_cn;
y = idu;

n = length(x); 
tau = 0;

for i = 1:n-1
    for j = i+1:n
        if (x(i) < x(j) && y(i) > y(j)) || (x(i) > x(j) && y(i) < y(j)) 
            tau = tau + 1; 
        end 
    end 
end 
tau_Hub = tau / (n*(n-1)/2)

x = rank_cn;
y = idv;

n = length(x); 
tau = 0;

for i = 1:n-1
    for j = i+1:n
        if (x(i) < x(j) && y(i) > y(j)) || (x(i) > x(j) && y(i) < y(j)) 
            tau = tau + 1; 
        end 
    end 
end 
tau_Authority = tau / (n*(n-1)/2)

% Authority rank is most similar to research ranking, followed by Hub
% ranking and PageRank.

D = sum(W,2);
n = length(D);
idnz = find(D>0);
T = zeros(n,n);
T(idnz,idnz) = diag(1./D(idnz)) * W(idnz,idnz);
alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];

for i=1:length(alpha)
	T1 = alpha(i) * T + (1-alpha(i))*ones(n,1)*ones(1,n)/n;
    [evec,eval] = eigs(T1',1);
	[score_page(:,i)]=evec/sum(evec);  % pagerank score
    [~,id]=sort(score_page(:,1),'descend');
    id_list(:,i) = id;
end

% We can see that the PageRank does not change with different alpha,
% therefore the ranking is stable.