 function [theta]=lcf_omp(y,A,t)
% y=Phi*x;
% x=Psi*theta;
% y=Phi*Psi*x;
% 令A=Phi*Psi 则y=A*theta;

[y_rows,y_colums]=size(y);
if y_rows<y_colums
    y=y';     % y应该是个列向量
end
[M,n]=size(A);             % A=m*n;
%% 开辟存储空间
theta=zeros(n,1);           % 存储
At=zeros(M,t);              %用来迭代过程中存储A被选择的列
Pos_theta=zeros(1,t);       %用来迭代过程中存储A被选择的列序号
r_n=y;                     % 初始化残差为y
%%
for ii=1:t
    product=A'*r_n;        %传感矩阵A各列与残差的内积
    [val,pos]=max(abs(product)); %找到最大内积绝对值，即与残差最相关的列
    At(:,ii)=A(:,pos);         %存储这一列
    Pos_theta(ii)=pos;           %储存这一列的序号
    A(:,pos)=zeros(M,1);         %清零A的这一列，其实此行可以删除，因为其与残差正交；
    theta_ls=((At(:,1:ii))'*At(:,1:ii))^(-1)*At(:,1:ii)'*y;    % 最小二乘解  y=A_t(:,1:i)*theta
    r_n=y-At(:,1:ii)*theta_ls;     % 更新残差
end
theta(Pos_theta)=theta_ls;         % 恢复出的theta

    
