
%CSIC HW-7 Q4
clear; clc;
data=load('karate.mat');
A=data.A; %Adjacency Matrix
d=sum(A,2);
D=diag(d); %Degree matrix
L=D-A; %Graph Laplacian

%Q4(a)
[evec,eval] = eig(L);
[e1,idx]=sort(diag(eval));
lambda2=e1(2);
f=evec(:,idx(2)); %second smalles eigenvector

%Q4(b)
[f1,idx1]=sort(f); %Fiedler vector
plot(f1,'.-');
%plotting Adjacency Matrix
figure;
spy(A);
figure;
spy(A(idx1,idx1));

%Q4(c)
med=median(f);
S_star=find(f<med);
S_star_bar=find(f>=med);

delta_S_star=sum(sum(A(S_star,S_star_bar)));
alpha_f=delta_S_star/(min(sum(d(S_star)),sum(d(S_star_bar)) ));

%Q4(d)
check_lambda=(lambda2>alpha_f);

%Q4(e)
S_plus=find(f<=0);
S_minus=find(f>0);

delta_S_plus=sum(sum(A(S_plus,S_minus)));
h_s_plus=delta_S_plus/(min(sum(d(S_plus)),sum(d(S_minus)) ));


%Q4(f) Recursive partition
A1=A(S_plus,S_plus);
e1=sum(A1,2);
E1=diag(e1); %Degree matrix
L1=E1-A1; %Graph Laplacian

[V1,D1] = eigs(L1, 2, 'SA');

S_plus_1=find(V1(:,2)<=0);
S_plus_2=find(V1(:,2)>0);

A2=A(S_plus,S_plus);
e2=sum(A2,2);
E2=diag(e2); %Degree matrix
L2=E2-A2; %Graph Laplacian
[V2,D2] = eigs(L2, 2, 'SA');

S_minus_1=find(V2(:,2)<=0);
S_minus_2=find(V2(:,2)>0);








