
clear; clc;
data=load('snp452-data');


X=data.X;
%Part (a)
Y=log(X');
a=size(Y);
%Part (a)
deltaY=zeros(a(1),a(2)-1);

for t=1:a(2)-1
    deltaY(:,t)=Y(:,t+1)-Y(:,t);
end

%part(c)
Sigma=(1/(a(2)-1))*(deltaY*deltaY');

%part(d)
[V,D] = eig(Sigma);
d=diag(D);
[lambdas,idx]=sort(d,'descend');
V1=V(:,idx);

%part(e) Horn's Parallel Analysis
R=500;
counts=zeros(size(lambdas));
shuffledY=zeros(a(1),a(2)-1);
for i=1:R
   
    for j=1:a(1)
        shuffledY(j,:)=deltaY(j,randperm(a(2)-1));
    end
    Sigma1=(1/(a(2)-1))*(shuffledY*shuffledY');
    e=eig(Sigma1);
    e1=sort(e,'descend');
    
    for k=1:size(counts)
       if(e1(k)>lambdas(k))
           counts(k)=counts(k)+1;
       end   
    end
    
    
end
pvals=(counts+1)/(R+1);


figure()
plot(pvals,'k-*');
xlabel('eigenvalue index')
ylabel('p-value')





 