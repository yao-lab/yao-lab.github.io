

% Wigner's semicircle law

clear; clc;
n=400;
W=zeros(n,n);

for j=1:n
    for i=1:j
     W(i,j)= sqrt(1/(4*n))*randn;     
     W(j,i)=W(i,j);   
    end
    
end

t=-1:2/1000:1;
wigner_law=(2/pi).*sqrt(1-t.^2);

evals=eig(W);

x0=min(evals)-1; binsize=0.1; xf=max(evals)+1;
xspan=[x0:binsize:xf]; 

figure()
h=hist(evals,xspan);          
hn=h/(length(evals)*binsize);
bar(xspan,hn);               
hhh = findobj(gca,'Type','patch');
set(hhh,'FaceColor','b','EdgeColor','w')

hold on
plot(t,wigner_law,'LineWidth',1.5)
title('Histogram of eigenvalues of W and Wigner law')
xlabel('\lambda')
ylabel('p(\lambda)')
legend('Histogram','Wigner Law')

%Part(b)


