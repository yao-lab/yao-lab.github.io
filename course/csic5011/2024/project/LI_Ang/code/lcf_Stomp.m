function [theta]=lcf_Stomp(y,A,S,ts)
% S ��StOMP ���������е�������
% ts����ֵ����
if nargin<4
    ts=2.5;               %ts��Χ[2,3]��Ĭ��ֵ2.5
end
if nargin<3
    S=10;                 %S Ĭ��ֵΪ10
end
[y_rows,y_columns]=size(y);
if y_rows<y_columns
    y=y';         % yӦ����������
end
[m,n]=size(y);
theta=zeros(n,1);
pos_num=[];       % �����洢���������б�Aѡ��������
r_n=y;            %��ʼ���в�Ϊr_n;
for i=1:S
    product=A'*r_n;            %A ������в���ڻ�
    sigma=norm(r_n)/sqrt(m);
    Js=find(abs(product)>ts*sigma);       %ѡ��������ֵ����
    Is=union(pos_num,Js);       %pos_num ��Js����
    if  length(pos_num)==length(Is)
        if i==1
            theta_ls=0;          %��ֹ��һ�ξ���������theta_ls�޶���
        end
        break; %���û���µ��б�ѡ�о�����ѭ��
    end
    % At������Ҫ������������Ϊ��С���˵Ļ������������޹أ�
    if length(Is)<=m
        pos_num=Is;   %��������ż���
        At=A(:,pos_num); %��A���⼸����ɾ���At
    else
        % At�����������������б�Ϊ������أ�At'*At ��������
        if i==1
            theta_ls=0;             %��ֹ��һ�ξ��˳�����theta_ls�޶���
        end
        break; % ����forѭ��
    end
    theta_ls=(At'*At)^(-1)*At'*y; %��С���˽�
    %At*theta_ls��y��At�пռ������ͶӰ
    r_n=y-At*theta_ls;            %���²в�
    if norm(r_n)<1e-6   % ѭ��ֱ��r=0;
        break; % ����forѭ��
    end
end
theta(pos_num)=theta_ls;  %�ָ�����theta
end
  
      