% 对于boost buck 电路的纹波电压进行CS感知采样。分别利用BP、LASSO、OMP、CosaMP、IHT进行重构
%% 加载数据
load('boost40k.mat')
i=796000:797023;
f=boost40k(i,2);
n=length(f);
m=double(int32(512));
noise=0.005*randn(1024,1);

x=f+noise;

%% 构造感知矩阵   高斯矩阵
randn('state',7);
Phi=sqrt(1/m)*randn(m,n);                      %Phi  感知矩阵
y=Phi*x;                                        % y   观测向量
Psi=inv(fft(eye(n,n)));                         % Psi 稀疏矩阵
% Psi=dct(eye(n,n));
A=Phi*Psi;                                      % A   测量矩阵
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








