 function [theta]=lcf_omp(y,A,t)
% y=Phi*x;
% x=Psi*theta;
% y=Phi*Psi*x;
% ��A=Phi*Psi ��y=A*theta;

[y_rows,y_colums]=size(y);
if y_rows<y_colums
    y=y';     % yӦ���Ǹ�������
end
[M,n]=size(A);             % A=m*n;
%% ���ٴ洢�ռ�
theta=zeros(n,1);           % �洢
At=zeros(M,t);              %�������������д洢A��ѡ�����
Pos_theta=zeros(1,t);       %�������������д洢A��ѡ��������
r_n=y;                     % ��ʼ���в�Ϊy
%%
for ii=1:t
    product=A'*r_n;        %���о���A������в���ڻ�
    [val,pos]=max(abs(product)); %�ҵ�����ڻ�����ֵ������в�����ص���
    At(:,ii)=A(:,pos);         %�洢��һ��
    Pos_theta(ii)=pos;           %������һ�е����
    A(:,pos)=zeros(M,1);         %����A����һ�У���ʵ���п���ɾ������Ϊ����в�������
    theta_ls=((At(:,1:ii))'*At(:,1:ii))^(-1)*At(:,1:ii)'*y;    % ��С���˽�  y=A_t(:,1:i)*theta
    r_n=y-At(:,1:ii)*theta_ls;     % ���²в�
end
theta(Pos_theta)=theta_ls;         % �ָ�����theta

    
