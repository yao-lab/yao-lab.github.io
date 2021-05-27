function y=minusentropy(p)
    load univ_cn.mat W_cn univ_cn rank_cn

    v = rank_cn;        % research rank of universities
    webpage = univ_cn;  % webpage of universities in mainland china
    W = W_cn;           % Link weight matrix
    %%%%add a link between every two univ
    W=W+ones(76)-eye(76);
    Wsum=sum(sum(W));
    y=0;%g=zeros(5776,1);
    for i=1:76
        for j=1:76
            if W(i,j)~=0
                y=y+p((i-1)*76+j)*log2(p((i-1)*76+j));%/(W(i,j)/Wsum));
         %       g((i-1)*76+j)=log2(p((i-1)*76+j))+1/log(2);
            end
        end
    end
end
