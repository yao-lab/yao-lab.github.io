load univ_cn.mat W_cn univ_cn rank_cn

v = rank_cn;        % research rank of universities
webpage = univ_cn;  % webpage of universities in mainland china
W = W_cn;           % Link weight matrix

%%%%add a link between every two univ
W=W+ones(76)-eye(76);

%Ap<=b
A=[];b=[];
%equality constraint matrix Aeq
Aeq=zeros(77,76*76);
Aeq(77,:)=ones(1,76*76);
for i=1:76
    for j=1:76
        if W(i,j)~=0
            Aeq(i,76*(i-1)+j)=1;
        end
        if W(j,i)~=0
            Aeq(i,76*(j-1)+i)=-1;
        end
    end
end
%equality constaint rhs b
beq=zeros(77,1);beq(77)=1;

% lb and ub of p
lb=zeros(76*76,1);
ub=zeros(76*76,1);
for i=1:76
    for j=1:76
        if W(i,j)~=0
            ub(76*(i-1)+j)=Inf;
        end
    end
end

% initial point p_0
p0=zeros(5776,1);
for i=1:76
    for j=1:76
        p0(j+76*(i-1))=1/3230;
    end
end
options = optimoptions('fmincon','MaxFunctionEvaluations',1e7,'Display','iter-detailed');%,'Algorithm','sqp');
[x,fval,exitflag,output,lambda] = fmincon('minusentropy',p0,A,b,Aeq+0*eye(77,5776),beq,lb,ub,[],options);
%traffic
for i=1:76
    traffic_rank(i)=sum(x(76*i-75:76*i));
end

[~,traffic_r_index]=sort(traffic_rank,'descend');


corr(traffic_r_index',rank_cn,'Type','Spearman')

corr(traffic_r_index',rank_cn,'Type','Kendall')

%temperature
for i=1:76
    temperature_rank(i)=1/lambda.eqlin(i);
end

[~,temperature_r_index]=sort(temperature_rank,'descend');


corr(temperature_r_index',rank_cn,'Type','Spearman')

corr(temperature_r_index',rank_cn,'Type','Kendall')