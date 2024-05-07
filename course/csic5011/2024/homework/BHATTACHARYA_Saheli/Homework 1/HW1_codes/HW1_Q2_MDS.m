% 	Delhi	Kolkata	Chennai	Mumbai	Bhopal	Bengaluru	Hyderabad	Agra
% Delhi	0	1305.6	1758.9	1157.3	596.9	1742	1256.5	179.2
% Kolkata	1305.6	0	1360.6	1657.6	1125.4	1559.8	1181.1	1163.6
% Chennai	1758.9	1360.6	0	1027.7	1172.9	286.8	516.5	1586.7
% Mumbai	1157.3	1657.6	1027.7	0	667.1	840.1	619.6	1049.9
% Bhopal	596.9	1125.4	1172.9	667.1	0	1145.1	662.8	439.8
% Bengaluru	1742	1559.8	286.8	840.1	1145.1	0	500.9	1581.1
% Hyderabad	1256.5	1181.1	516.5	619.6	662.8	500.9	0	1089.8
% Agra	179.2	1163.6	1586.7	1049.9	439.8	1581.1	1089.8	0
clear; clc;
f1=load('cities.mat');
D=f1.D.^2;   %Squared Distance Matrix
n=length(D);
k=3; %Euclidean space of dimensionm 2
city ={'Delhi','Kolkata','Chennai','Mumbai','Bhopal','Bengaluru','Hyderabad','Agra'};

H=eye(n)-(1/n)*ones(n,1)*ones(1,n);

B=(-1/2)*H*D*H';

[U,Lambdam]=eig(B);

lamdas=diag(Lambdam);
[lamdas_sorted, idx]=sort(lamdas,'descend');

U1=U(:,idx);
U2=U1(:,1:k);
Xk=U2*sqrtm(diag(lamdas_sorted(1:k)));
c = linspace(1,20,n);
sz=40;

figure(1);
plot(lamdas_sorted/sum(lamdas));
title('Normalized Eigenvalues of B')

figure(2);
scatter3(Xk(:,1),Xk(:,2),Xk(:,3),sz,c,'filled');
text(Xk(:,1)+25,Xk(:,2),Xk(:,3),city)
% xlabel('Km')
% ylabel('Km')