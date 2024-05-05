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
    x=x';             % x应该是一个列向量
end
n=size(Phi,2);
y=zeros(n,1);         %初始化y=0;
loop=0;
while(norm(x-Phi*y)>epsilon && loop<loopmax)
    y=y+Phi'*(x-Phi*y)*mu;% 更新y
    % 下面两行代码实现了H_M(,)
    % 按降序排列y的绝对值
    [ysorted inds]=sort(abs(y),'descend');
    % 将除M外的所有最大坐标设为0
    y(inds(M+1:n))=0;
    loop=loop+1;
end
end
    