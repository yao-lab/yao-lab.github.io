load univ_cn.mat W_cn univ_cn rank_cn

v = rank_cn;        % research rank of universities
webpage = univ_cn;  % webpage of universities in mainland china
W = W_cn;           % Link weight matrix

D = sum(W,2);
n = length(D);
idnz = find(D>0);
T = zeros(n,n);
T(idnz,idnz) = diag(1./D(idnz)) * W(idnz,idnz);

alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.85 0.9]; % alpha=0.85 is Google's PageRank choice

for i=1:length(alpha),
	T1 = alpha(i) * T + (1-alpha(i))*ones(n,1)*ones(1,n)/n;
    [evec,eval] = eigs(T1',1);
	[score_page(:,i)]=evec/sum(evec);  % pagerank score
    [~,idp(:,i)]=sort(score_page(:,i),'descend');

end

% PageRank
%[~,idp]=sort(score_page(:,8),'descend');

%webpage{idp(1:50)}

%spearman
for i=1:9
    S_rank(i)=corr(idp(:,i),rank_cn,'Type','Spearman');
end

for i=1:9
    for j=1:9
        S_alpha(i,j)=corr(idp(:,i),idp(:,j),'Type','Spearman');
    end
end

%kendall
for i=1:9
    K_rank(i)= corr(idp(:,i),rank_cn,'Type','Kendall');
end

for i=1:9
    for j=1:9
        K_alpha(i,j)=corr(idp(:,i),idp(:,j),'Type','Kendall');
    end
end