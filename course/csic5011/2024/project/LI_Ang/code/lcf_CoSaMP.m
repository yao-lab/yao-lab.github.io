function [ theta ] = lcf_CoSaMP( y,A,K )
%   CS_CoSaOMP
%   Detailed explanation goes here
%   y = Phi * x
%   x = Psi * theta
%    y = Phi*Psi * theta
%   �� A = Phi*Psi, ��y=A*theta
%   K is the sparsity level
%   ������֪y��A����theta
%   Reference:Needell D��Tropp J A��CoSaMP��Iterative signal recovery from
%   incomplete and inaccurate samples[J]��Applied and Computation Harmonic 
%   Analysis��2009��26��301-321.
    [m,n] = size(y);
    if m<n
        y = y'; %y should be a column vector
    end
    [M,N] = size(A); %���о���AΪM*N����
    theta = zeros(N,1); %�����洢�ָ���theta(������)
    pos_num = []; %�������������д洢A��ѡ��������
    res = y; %��ʼ���в�(residual)Ϊy
    for kk=1:K %������K��
        %(1) Identification
        product = A'*res; %���о���A������в���ڻ�
        [val,pos]=sort(abs(product),'descend');
        Js = pos(1:2*K); %ѡ���ڻ�ֵ����2K��
        %(2) Support Merger
        Is = union(pos_num,Js); %Pos_theta��Js����
        %(3) Estimation
        %At������Ҫ������������Ϊ��С���˵Ļ���(�������޹�)
        if length(Is)<=M
            At = A(:,Is); %��A���⼸����ɾ���At
        else %At�����������������б�Ϊ������ص�,At'*At��������
            if kk == 1
                theta_ls = 0;
            end
            break; %����forѭ��
        end
        %y=At*theta��������theta����С���˽�(Least Square)
        theta_ls = (At'*At)^(-1)*At'*y; %��С���˽�
        %(4) Pruning
        [val,pos]=sort(abs(theta_ls),'descend');
        %(5) Sample Update
        pos_num = Is(pos(1:K));
        theta_ls = theta_ls(pos(1:K));
        %At(:,pos(1:K))*theta_ls��y��At(:,pos(1:K))�пռ��ϵ�����ͶӰ
        res = y - At(:,pos(1:K))*theta_ls; %���²в� 
        if norm(res)<1e-6 %Repeat the steps until r=0
            break; %����forѭ��
        end
    end
    theta(pos_num)=theta_ls; %�ָ�����theta
end