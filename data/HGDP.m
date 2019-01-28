load('/data/hgdp/data.mat');
load('HGDP_region.mat');
load('Personal_Info');

%% PCA  %%
M=X(ind1,ind2);
Pop7Groups=Pop7Groups(ind1);
Population=Population(ind1);
id=id(ind1);
[n,p]=size(M);

tab=tabulate(Population);
N_Pop=cell2mat(tab(:,2));
id_Pop=[];
for i=1:length(N_Pop)
    id_Pop=[id_Pop,i*ones(1,N_Pop(i))];
end

Mbar=M-ones(n,1)*mean(M);
Sigma=Mbar*Mbar';
[U,L]=eig(Sigma);
[l,I2]=sort(diag(L),'descend');
color={'bo','go','ro','co','mo','yo','ko','b.','g.','r.','c.','m.','y.','k.'};
i=1;
k=1;
for j=1:51
    a=1:n;
    a=a(id_Pop==j);
    if (id(a(1)) > i)
        %legend boxoff
        xlabel('PC1'); 
        ylabel('PC2');
        title(Pop7Groups(a(1)-1));
        hold off;
        legend(lgd);
        print(gcf,'-depsc',strcat(cell2mat(Pop7Groups(a(1)-1)),'.eps'));
        i=i+1;
        k=1;
        clear lgd;
    end
    plot(U(a,I2(1)),-U(a,I2(2)), cell2mat(color(mod(j-1,14)+1)));hold on;
    lgd(k)=Population(a(1));k=k+1;
end
xlabel('PC1'); 
ylabel('PC2');
title(Pop7Groups(a(1)));
hold off;
legend(lgd);
print(gcf,'-depsc',strcat(cell2mat(Pop7Groups(a(1)-1)),'.eps'));
        
k=1;        
for i=1:7
    a=1:n;
    a=a(id==i);
    T=M(a,:);
    Tbar=T-ones(length(a),1)*mean(T);
    Sigma_T=Tbar*Tbar';
    [U_T,L_T]=eig(Sigma_T);
    [l_T,I_T]=sort(diag(L_T),'descend');
    for j=1:51
        b=1:length(a);
        b=b(id_Pop(a)==j);
        if length(b)>0
            plot(U_T(b,I_T(1)),U_T(b,I_T(2)), cell2mat(color(mod(j-1,14)+1)));hold on;
        lgd(k)=Population(a(b(1)));k=k+1;
        end
    end
    xlabel('PC1'); 
    ylabel('PC2');
    title(Pop7Groups(a(1)));
    hold off;
    legend(lgd);
    clear lgd;
    print(gcf,'-depsc',strcat(cell2mat(Pop7Groups(a(1))),'_sub.eps'));    
    k=1;
end


for i=1:7
    a=1:n;
    a=a(id==i);
    plot(U(a,I2(1)),-U(a,I2(2)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
title('PCA 2D');
hold off;
print(gcf,'-depsc','PCA_2D.eps');
for i=1:7
    a=1:n;
    a=a(id==i);
    plot3(U(a,I2(1)),-U(a,I2(2)),-U(a,I2(3)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('PCA 3D');
hold off;
print(gcf,'-depsc','PCA_3D.eps');


%% MDS %%

A0=double(X(ind1,:)==0);
A1=double(X(ind1,:)==1);
A2=double(X(ind1,:)==2);
D=(A0+A2)*A1';
C=A2*A0';
D1=D+C;
D1=D1+D1';
D2=D1+C;
D2=D2+D2';
D3_sq=D2+2*C;
D1_sq=D1.^2;
D2_sq=D2.^2;

H=eye(n)-ones(n,n)/n;
B1=-H*D1_sq*H/2;
[V,lambda]=eig(B1);
[lam,I]=sort(diag(lambda),'descend');

i=1;
k=1;
for j=1:51
    a=1:n;
    a=a(id_Pop==j);
    if (id(a(1)) > i)
        %legend boxoff
        xlabel('PC1'); 
        ylabel('PC2');
        title(Pop7Groups(a(1)-1));
        hold off;
        legend(lgd);
        print(gcf,'-depsc',strcat(cell2mat(Pop7Groups(a(1)-1)),'_MDS.eps'));
        i=i+1;
        k=1;
        clear lgd;
    end
    plot(V(a,I(1)),-V(a,I(2)), cell2mat(color(mod(j-1,14)+1)));hold on;
    lgd(k)=Population(a(1));k=k+1;
end
xlabel('PC1'); 
ylabel('PC2');
title(Pop7Groups(a(1)));
hold off;
legend(lgd);
print(gcf,'-depsc',strcat(cell2mat(Pop7Groups(a(1)-1)),'_MDS.eps'));
        
k=1;        
for i=1:7
    a=1:n;
    a=a(id==i);
    T1=D1_sq(a,a);
    n_T=length(a);
    H_T=eye(n_T)-ones(n_T,n_T)/n_T;
    B_T=-H_T*T1*H_T/2;
    [V_T,lambda_T]=eig(B_T);
    [lam_T,I_T]=sort(diag(lambda_T),'descend');
    for j=1:51
        b=1:length(a);
        b=b(id_Pop(a)==j);
        if length(b)>0
            plot(V_T(b,I_T(1)),V_T(b,I_T(2)), cell2mat(color(mod(j-1,14)+1)));hold on;
        lgd(k)=Population(a(b(1)));k=k+1;
        end
    end
    xlabel('PC1'); 
    ylabel('PC2');
    title(Pop7Groups(a(1)));
    hold off;
    legend(lgd);
    clear lgd;
    print(gcf,'-depsc',strcat(cell2mat(Pop7Groups(a(1))),'_MDS_sub.eps'));    
    k=1;
end


for i=1:7
    a=1:n;
    a=a(id==i);
    plot(V(a,I(1)),-V(a,I(2)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
title('PCA 2D');
hold off;
print(gcf,'-depsc','MDS_2D.eps');
for i=1:7
    a=1:n;
    a=a(id==i);
    plot3(V(a,I(1)),-V(a,I(2)),-V(a,I(3)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('PCA 3D');
hold off;
print(gcf,'-depsc','MDS_3D.eps');


%% Fst %%
Fst=zeros(51);
p0=zeros(51,p);
p1=zeros(51,p);
p1=zeros(51,p);
alpha=zeros(51,p);
a=1:n;
for i=1:51
    ai=a(id_Pop==i);
    ni=N_Pop(i);
    p0(i,:)=sum(A0(ai,:))/ni;
    p1(i,:)=sum(A1(ai,:))/ni;
    p2(i,:)=sum(A2(ai,:))/ni;
    alpha(i,:)=1-p0(i,:).^2-p1(i,:).^2-p2(i,:).^2;
end

for i=1:50
    for j=(i+1):51
        ni=N_Pop(i);
        nj=N_Pop(j);
        H_w=(ni*(p0(i,:)*(1-p0(i,:))'+p1(i,:)*(1-p1(i,:))'+p2(i,:)*(1-p2(i,:))')+...
            nj*(p0(j,:)*(1-p0(j,:))'+p1(j,:)*(1-p1(j,:))'+p2(j,:)*(1-p2(j,:))'))/p/(ni+nj);
        H_b=1-(p0(i,:)*p0(j,:)'+p1(i,:)*p1(j,:)'+p2(i,:)*p2(j,:)')/p;
        Fst(i,j)=1-H_w/H_b;
    end
end
Fst=Fst+Fst';
H=eye(51)-ones(51,51)/51;
B=-H*Fst.^2*H/2;
[W,lambda]=eig(B);
[lam,I]=sort(diag(lambda),'descend');
for i=1:7
    a=1:n;
    a=a(id==i);
    b=id_Pop(a);
    b=min(b):max(b);
    plot(W(b,I(1)),W(b,I(2)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
title('Fst');
hold off;
print(gcf,'-depsc','Fst_population.eps');
for i=1:7
    a=1:n;
    a=a(id==i);
    b=id_Pop(a);
    b=min(b):max(b);
    plot3(W(b,I(1)),W(b,I(2)),-W(b,I(3)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('Fst 3D');
hold off;
print(gcf,'-depsc','Fst_population_3D.eps');

save 'Fst.mat' Fst


%% Coancestry Coeffient %%
cc=zeros(51);
for i=1:50
    for j=(i+1):51
        ni=N_Pop(i);
        nj=N_Pop(j);
        temp=((p0(i,:)-p0(j,:)).^2+(p1(i,:)-p1(j,:)).^2+(p2(i,:)-p2(j,:)).^2)/2;
        ali=temp-(ni+nj)/4/ni/nj/(ni+nj-1)*(ni*alpha(i,:)+nj*alpha(j,:));
        ali_plus_bli=temp+(4*ni*nj-ni-nj)/4/ni/nj/(ni+nj-1)*(ni*alpha(i,:)+nj*alpha(j,:));
        %theta=ali./ali_plus_bli;
        cc(i,j)=sum(ali)/sum(ali_plus_bli);
    end
end
cc=cc+cc';
H=eye(51)-ones(51,51)/51;
B=-H*cc.^2*H/2;
[W,lambda]=eig(B);
[lam,I]=sort(diag(lambda),'descend');
for i=1:7
    a=1:n;
    a=a(id==i);
    b=id_Pop(a);
    b=min(b):max(b);
    plot(W(b,I(1)),W(b,I(2)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
title('CC');
hold off;
print(gcf,'-depsc','CC_population.eps');

for i=1:7
    a=1:n;
    a=a(id==i);
    b=id_Pop(a);
    b=min(b):max(b);
    plot3(W(b,I(1)),W(b,I(2)),-W(b,I(3)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('CC 3D');
hold off;
print(gcf,'-depsc','CC_population_3D.eps');

save 'CC.mat' cc


%% Add  %%
PCA=zeros(51);
a=1:n;
for i=1:50
    for j=(i+1):51
        ai=a(id_Pop==i);
        aj=a(id_Pop==j);
        %N_within=(sum(sum(D1(ai,ai)))+sum(sum(D1(ai,ai))))/...
        %    ((N_Pop(i)-1)*N_Pop(i)+(N_Pop(j)-1)*N_Pop(j));
        N_between=sum(sum(D1(ai,aj)))/N_Pop(j)/N_Pop(i);
        PCA(i,j)=N_between;%1-N_within/N_between;
    end
end
PCA=PCA+PCA';
H=eye(51)-ones(51,51)/51;
B=-H*PCA.^2*H/2;
[W,lambda]=eig(B);
[lam,I]=sort(diag(lambda),'descend');
for i=1:7
    a=1:n;
    a=a(id==i);
    b=id_Pop(a);
    b=min(b):max(b);
    plot(W(b,I(1)),W(b,I(2)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
title('PCA');
hold off;
print(gcf,'-depsc','PCA_population.eps');

for i=1:7
    a=1:n;
    a=a(id==i);
    b=id_Pop(a);
    b=min(b):max(b);
    plot3(W(b,I(1)),W(b,I(2)),W(b,I(3)),cell2mat(color(mod(i-1,14)+1)));hold on;
end
legend('   Africa','   America','   Central South Asia','   Est Asia',...
'   Europe','   Middle Est','   Oceania')
legend boxoff
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('PCA 3D');
hold off;
print(gcf,'-depsc','PCA_population_3D.eps');
