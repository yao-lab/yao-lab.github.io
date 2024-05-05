% ����boost buck ��·���Ʋ���ѹ����CS��֪�������ֱ�����BP��LASSO��OMP��CosaMP��IHT�����ع�
%% ��������
load('boost40k.mat')
i=796000:797023;
f=boost40k(i,2);
n=length(f);
m=double(int32(512));
noise=0.005*randn(1024,1);

x=f+noise;

%% �����֪����   ��˹����
randn('state',7);
Phi=sqrt(1/m)*randn(m,n);                      %Phi  ��֪����
y=Phi*x;                                        % y   �۲�����
Psi=inv(fft(eye(n,n)));                         % Psi ϡ�����
% Psi=dct(eye(n,n));
A=Phi*Psi;                                      % A   ��������
%%  OMP
K=35;
theta=lcf_CoSaMP(y,A,K);
x_r=real(ifft(full(theta)));
% x_r=Psi*theta;
plot(x_r)
hold on
plot(f)
% %% BP
% M=79;mu=0.35;epsilon=1e-20;loopmax=3000;
% theta=lcf_IHT(y,A,M,mu,epsilon,loopmax);
% x_r=Psi*theta;
% plot(x_r)








