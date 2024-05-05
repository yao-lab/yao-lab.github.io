function [alpha]=lcf_linprog(y,Phi)
% Atomic decomposition by basis pursuit
% Detailed explanation goes here
% s=Phi*alpha(alpha is a sparse vector)
% Given s & Phi ,try to derive alpha
[s_rows,s_columns]=size(y);
if s_rows<s_columns
    y=y'; %s应该是列向量
end
p=size(Phi,2);
% 
c=ones(2*p,1);
A=[Phi,-Phi];
b=y;
lb=zeros(2*p,1);
x0=linprog(c,[],[],A,b,lb);
alpha=x0(1:p)-x0(p+1:2*p);
end
