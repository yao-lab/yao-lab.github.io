%Constructing B and theta
k=5;
n=300;
N=k*n;
B=csvread('B_sym.csv');
%theta=rand(N,1);
%for i=1:k
%   theta(((i-1)*n+1):(i)*n)=theta(((i-1)*n+1):(i)*n)*n/sum(theta(((i-1)*n+1):(i)*n));
%end
theta=ones(N,1);
    
%Checkin A
warning=0;
for i=1:k
    for j=1:k
        for r=1:n
            for s=1:n
                if(theta((i-1)*n+r)*theta((j-1)*n+s)*B(i,j)>1)
                    warning=warning+1;
                end
            end
        end
    end
end




%Constructing DCSBM
[A,labels]=DCSBM(n,B,theta);
m=size(A,1);
for i=1:m
    for j=1:m
        if(isnan(A(i,j)))
            A(i,j)=0;
        end
    end
end

d=zeros(1,m);
for i=1:m
   d(i)=sum(A(i,:));
end
D=diag(d);

%algorithm 1
[evec,eval]=eig(D-A,D);
v1=D^(-1/2)*evec(:,m-k+1:m);
[cidx, ctrs] = kmeans(v1,k);

%algorithm 2
L=D^(-1/2)*(D-A)*D^(-1/2);
[evec1,eval1]=eig(L);
v2=evec(:,1:k);
for i=1:m
    v2(i,:)=v2(i,:)/norm(v2(i,:),2);
end
[cidx1, ctrs1] = kmeans(v2,k);

%Computing Normalized Mutual Information
I1=NormalizedMI(labels,cidx);
I2=NormalizedMI(labels,cidx1);
