function [y]=lcf_IHT(x,Phi,M,mu,epsilon,loopmax)
if nargin<6
    loopmax=3000;
end
if nargin<5
    epsilon=1e-3;
end
if nargin<4
    mu=1;
end
[x_rows,x_columns]=size(x);
if x_rows<x_columns
    x=x';             % xӦ����һ��������
end
n=size(Phi,2);
y=zeros(n,1);         %��ʼ��y=0;
loop=0;
while(norm(x-Phi*y)>epsilon && loop<loopmax)
    y=y+Phi'*(x-Phi*y)*mu;% ����y
    % �������д���ʵ����H_M(,)
    % ����������y�ľ���ֵ
    [ysorted inds]=sort(abs(y),'descend');
    % ����M����������������Ϊ0
    y(inds(M+1:n))=0;
    loop=loop+1;
end
end
    