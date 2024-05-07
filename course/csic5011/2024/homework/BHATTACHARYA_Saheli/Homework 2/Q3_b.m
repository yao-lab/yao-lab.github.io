
clear; clc;
n=400;
W=zeros(n,n);
sigma=1;

for j=1:n
    for i=1:j
     W(i,j)= sqrt(sigma/n)*randn;     
     W(j,i)=W(i,j);   
    end
    
end

u=zeros(n,1);
u(1)=1;

lambda0=0:(2*sigma)/10:2*sigma;
true_lambda=zeros(1,length(lambda0));
theo_lambda=zeros(1,length(lambda0));
true_u_t_v=zeros(1,length(lambda0));
theo_u_t_v=zeros(1,length(lambda0));

for i=1:length(lambda0)
   W1=W+lambda0(i)*(u*u');
   [V,D] = eig(W1);
   d=diag(D);
   [emax,idx]=max(d);
   
   true_lambda(i)=emax;
   theo_lambda(i)=lambda0(i)+(1/lambda0(i));
   
   true_u_t_v(i)=(u'*V(:,idx)).^2;
   theo_u_t_v(i)=(1-(1/(lambda0(i))^2));
end


figure(1)
plot(lambda0,true_lambda,'r-o')
hold on
plot(lambda0,theo_lambda,'k--*')
hold off
legend('True- \lambda', 'Theoretical- \lambda')
xlabel('SNR')
ylabel('\lambda')
xticks([0 sigma/2 sigma 2*sigma])
xticklabels({'0','\sigma/2','\sigma','2 \sigma','interpreter','latex'})
title('Plot of \lambda vs SNR (\sigma=1)')

figure(2)
plot(lambda0,true_u_t_v,'r-o')
hold on
plot(lambda0,theo_u_t_v,'k--*')
hold off
legend('True- <u,v>^2', 'Theoretical- <u,v>^2')
xlabel('SNR')
ylabel('<u,v>^2')
xticks([0 sigma/2 sigma 2*sigma])
xticklabels({'0','\sigma/2','\sigma','2 \sigma','interpreter','latex'})

title('Plot of <u,v>^2 vs SNR (\sigma=1) ')




