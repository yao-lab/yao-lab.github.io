clear
clc
n = 1000;
V1 = randn(n,1)*sqrt(290);
V2 = randn(n,1)*sqrt(300);
V3 = -0.3*V1+0.925*V2+randn(n,1);

X1 = V1+randn(n,1);
X2 = V1+randn(n,1);
X3 = V1+randn(n,1);
X4 = V1+randn(n,1);
X5 = V2+randn(n,1);
X6 = V2+randn(n,1);
X7 = V2+randn(n,1);
X8 = V2+randn(n,1);
X9 = V3+randn(n,1);
X10 = V3+randn(n,1);

X = [X1 X2 X3 X4 X5 X6 X7 X8 X9 X10];% n by 10 matrix

sample_covariance = cov(X,1); % sample covariance matrix

%true covariance matrix
covariance = zeros(10,10);
for i=1:4
    for j=1:4
        if i==j
            covariance(i,j)=291;
        else
            covariance(i,j)=290;
        end
    end
end
for i=5:8
    for j=1:4
        covariance(i,j)=0;
        covariance(j,i)=0;
    end
end
for i=5:8
    for j=5:8
        if i==j
            covariance(i,j)=301;
        else
            covariance(i,j)=300;
        end
    end
end
for i=1:4
    for j=9:10
        covariance(i,j)=-0.3*290;
        covariance(j,i)=-0.3*290;
    end
end
for i=5:8
    for j=9:10
        covariance(i,j)=0.925*300;
        covariance(j,i)=0.925*300;
    end
end
covariance(9,9)=284.7875;covariance(10,10)=0.3^2*290+0.925^2*300+1+1;
covariance(9,10)=283.7875;covariance(10,9)=0.3^2*290+0.925^2*300+1;
%end of computation of true covariance matrix

[U, S, V] = svd(covariance); %A=U*S*V', S in descending order
eigenvalues = diag(S);
%disp('the top 4 principal components of true covariance matrix are')
%disp(eigenvalues(1:4))


d = 10;
e = ones(d,1);

%first step
lambda = 0;

cvx_begin
variable C(d,d) symmetric;
C == semidefinite(d);
maximize(trace(covariance*C)-lambda*(e'*abs(C)*e));
subject to
trace(C) == 1;
cvx_end

target = trace(covariance*C)-lambda*(e'*abs(C)*e);

%second step
cov2 = covariance - C;
lambda = 94.327;

cvx_begin
variable C2(d,d) symmetric;
C2 == semidefinite(d);
maximize(trace(cov2*C2)-lambda*(e'*abs(C2)*e));
subject to
trace(C2) == 1;
cvx_end

target2 = trace(cov2*C2)-lambda*(e'*abs(C2)*e);

%third step
cov3 = cov2 - C2;
figure
for lambda = 299.329:0.0001:299.331
    cvx_begin
    variable C3(d,d) symmetric;
    C3 == semidefinite(d);
    maximize(trace(cov3*C3)-lambda*(e'*abs(C3)*e));
    subject to
    trace(C3) == 1;
    cvx_end
    target3 = trace(cov3*C3)-lambda*(e'*abs(C3)*e);
    plot(lambda,target3,'ro',lambda,eigenvalues(3),'g+')
    hold on
end
hold off
xlabel('\lambda')
ylabel('3rd largest singular value')
legend('spca sval','true sval')
title('\lambda with 3rd spca sval and true sval')
%we can see that lambda=299.33 is the best for the third singular value 
%based on first lambda=0 and second lambda=94.327