function [theta]=lcf_Stomp(y,A,S,ts)
% S 是StOMP 迭代过程中的最大次数
% ts是阈值参数
if nargin<4
    ts=2.5;               %ts范围[2,3]，默认值2.5
end
if nargin<3
    S=10;                 %S 默认值为10
end
[y_rows,y_columns]=size(y);
if y_rows<y_columns
    y=y';         % y应该是列向量
end
[m,n]=size(y);
theta=zeros(n,1);
pos_num=[];       % 用来存储迭代过程中被A选择的列序号
r_n=y;            %初始化残差为r_n;
for i=1:S
    product=A'*r_n;            %A 各列与残差的内积
    sigma=norm(r_n)/sqrt(m);
    Js=find(abs(product)>ts*sigma);       %选出大于阈值的列
    Is=union(pos_num,Js);       %pos_num 与Js并集
    if  length(pos_num)==length(Is)
        if i==1
            theta_ls=0;          %防止第一次就跳出导致theta_ls无定义
        end
        break; %如果没有新的列被选中就跳出循环
    end
    % At的行数要大于列数，此为最小二乘的基础（列线性无关）
    if length(Is)<=m
        pos_num=Is;   %更新列序号集合
        At=A(:,pos_num); %将A的这几列组成矩阵At
    else
        % At的列数大于行数，列必为线性相关，At'*At 将不可逆
        if i==1
            theta_ls=0;             %防止第一次就退出导致theta_ls无定义
        end
        break; % 跳出for循环
    end
    theta_ls=(At'*At)^(-1)*At'*y; %最小二乘解
    %At*theta_ls是y在At列空间的正交投影
    r_n=y-At*theta_ls;            %更新残差
    if norm(r_n)<1e-6   % 循环直到r=0;
        break; % 跳出for循环
    end
end
theta(pos_num)=theta_ls;  %恢复出的theta
end
  
      