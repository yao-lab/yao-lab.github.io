%CSIC:5011, HW-5, Q2(a)
clear; clc;
n=1000000;
d=10;
X=zeros(d,n);

for i=1:n
   v1=sqrt(290)*randn;
   v2=sqrt(300)*randn;
   v3=-0.3*v1+0.925*v2+randn;
   X(1:4,i)=v1*ones(4,1)+randn(4,1);
   X(5:8,i)=v2*ones(4,1)+randn(4,1);
   X(9:10,i)=v3*ones(2,1)+randn(2,1); 
end

Xc=X-mean(X,2);

Sigma=(1/n)*(Xc*Xc'); %Sample covariance matrix

Sigma_th= zeros(10,10); %theoretical covariance matrix

for i=1:4
    Sigma_th(i,i)=291;
end

for i=5:8
    Sigma_th(i,i)=301;
end

for i=9:10
    Sigma_th(i,i)=284.78;
end

for i=1:4
    for j=1:4
        if(i~=j)
           Sigma_th(i,j)=290;
        end
        
    end
    
end

for i=5:8
    for j=5:8
        if(i~=j)
           Sigma_th(i,j)=300;
        end
        
    end
    
end

for i=9:10
    for j=9:10
        if(i~=j)
           Sigma_th(i,j)=283.78;
           Sigma_th(j,i)=283.78;
        end
        
    end
    
end

for i=1:4
    for j=9:10
      Sigma_th(i,j)=-87;
      Sigma_th(j,i)=-87;
    end
    
end

for i=5:8
    for j=9:10
      Sigma_th(i,j)=277.5;
      Sigma_th(j,i)=277.5;
    end
    
end

%Q2(b)
[V,D]=eig(Sigma_th);
[d1,idx]=sort(diag(D),'descend');
U=V(:,idx(1:4)); %Top-4 eigenvectors

%Q2(c)

lambda = 20;
e = ones(d,1);

cvx_begin
    variable Y(d,d) symmetric;
    Y == semidefinite(d);
    minimize(-trace(Sigma_th*Y)+lambda*(e'*abs(Y)*e));
    subject to
        trace(Y)==1;
cvx_end

err=norm(U(:,1)-Y(:,1));

%Q2(d)
sigma1=Sigma_th-d1(1)*Y(:,1)*Y(:,1)';

cvx_begin
    variable Y1(d,d) symmetric;
    Y1 == semidefinite(d);
    minimize(-trace(sigma1*Y1)+lambda*(e'*abs(Y1)*e));
    subject to
        trace(Y1)==1;
cvx_end
err1=norm(U(:,2)-Y1(:,1));

%Q2(e)
sigma2=sigma1-d1(2)*Y1(:,1)*Y1(:,1)';

cvx_begin
    variable Y2(d,d) symmetric;
    Y1 == semidefinite(d);
    minimize(-trace(sigma2*Y2)+lambda*(e'*abs(Y2)*e));
    subject to
        trace(Y2)==1;
cvx_end
err2=norm(U(:,3)-Y2(:,1));


sigma3=sigma2-d1(3)*Y2(:,1)*Y2(:,1)';

cvx_begin
    variable Y3(d,d) symmetric;
    Y3 == semidefinite(d);
    minimize(-trace(sigma3*Y3)+lambda*(e'*abs(Y3)*e));
    subject to
        trace(Y3)==1;
cvx_end

err3=norm(U(:,4)-Y3(:,1));

