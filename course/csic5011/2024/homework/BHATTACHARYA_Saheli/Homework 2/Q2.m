
clear; clc;

gamma1=0.5;
sigma=1;
p=100;
n=p/gamma1;
lambda0=sqrt(gamma1)-gamma1:(6*gamma1)/10:sqrt(gamma1)+(5*gamma1);
% lambda0=sqrt(gamma1)-1:(6)/10:sqrt(gamma1)+5;

true_lambda=zeros(1,length(lambda0));
theo_lambda=zeros(1,length(lambda0));

true_u_t_v=zeros(1,length(lambda0));
theo_u_t_v=zeros(1,length(lambda0));
uut=zeros(p,p);
uut(1,1)=1;
u=zeros(p,1);
u(1)=1;

for i=1:length(lambda0)
    Sn=0;
    for j=1:n
       xi=sqrtm(eye(p)+lambda0(i)*uut)*randn(p,1);
       Sn=Sn+(xi*xi');
    end
    Sn=Sn/n;
    [V,D] = eig(Sn);
    d=diag(D);
    [emax,idx]=max(d);
    true_lambda(i)=emax;
    theo_lambda(i)=(1+lambda0(i))*(1+(gamma1/lambda0(i)));
    true_u_t_v(i)=(u'*V(:,idx)).^2;
    theo_u_t_v(i)=(1-(gamma1/(lambda0(i))^2))/(1-gamma1/lambda0(i));
end

figure(1)
plot(lambda0,true_lambda,'r-o')
hold on
plot(lambda0,theo_lambda,'k--*')
hold off
legend('True- \lambda', 'Theoretical- \lambda')
xlabel('SNR')
ylabel('\lambda')
% xticks([sqrt(gamma)-gamma sqrt(gamma) sqrt(gamma)+gamma sqrt(gamma)+5*gamma])
% xticklabels({'$\sqrt(\gamma)-\gamma$','$\sqrt(\gamma)$','$\sqrt(\gamma)+\gamma$','$\sqrt(\gamma)+5\gamma$','interpreter','latex'})

xticks([sqrt(gamma1)])
xticklabels({'\sqrt{\gamma}'})

title('Plot of SNR vs \lambda (\gamma=0.5)')

figure(2)
plot(lambda0,true_u_t_v,'r-o')
hold on
plot(lambda0,theo_u_t_v,'k--*')
hold off
legend('True- <u,v>^2', 'Theoretical- <u,v>^2')
xlabel('SNR')
ylabel('<u,v>^2')
% xticks([sqrt(gamma)-gamma sqrt(gamma) sqrt(gamma)+gamma sqrt(gamma)+5*gamma])
% xticklabels({'\sqrt(\gamma)-\gamma','\sqrt(\gamma)','\sqrt(\gamma)+\gamma','\sqrt(\gamma)+5*\gamma'})
xticks([sqrt(gamma1)])
xticklabels({'\sqrt{\gamma}'})
title('Plot of SNR vs <u,v>^2 (\gamma=0.5)')


