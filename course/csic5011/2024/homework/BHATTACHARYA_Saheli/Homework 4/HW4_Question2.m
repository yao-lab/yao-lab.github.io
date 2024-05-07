%HW-3 Q2
clear; clc;
d=20;
Niter=50;
% n=1:d;
% k=1:d;
prob=zeros(d,d);

for i=1:d %for k
   for j=1:d  %for n
       
     for k=1:Niter
        A=randn(j,d);
        xo=zeros(d,1);
        idx=randperm(d,i);
        xo(idx,1)=2*(rand(i,1)>.5) - 1;
%         for t=1:length(idx)
%         xo(idx(t),1)= 2*(rand>.5) - 1;
%         end
        b=A*xo;
        cvx_begin
          variable x(d)
          minimize(norm(x,1))
          subject to
          A*x==b;
        cvx_end
         if (norm(x-xo)<1e-3)
            prob(i,j)=prob(i,j)+1; 
         end
         
     end
       
       
   end
    
    
    
    
end


prob=prob/Niter;

figure(1)
imagesc(prob');
colorbar
xlabel('k')
ylabel('n')
title('Heat Map of p(n,k)')
