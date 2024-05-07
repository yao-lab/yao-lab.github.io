clear
clc
rng(42)
p = 800;
n = 800;

lam0 = 2:2:16;
k = length(lam0);

A = zeros(4,k);
for i=1:k
    [lam_real,lam_es,uv_real,uv_es] = RM(p,n,lam0(i));
    A(:,i) = [lam_real,lam_es,uv_real,uv_es]';
end

figure(1)
scatter(A(1,:),A(2,:))
hold on
plot(3:20,3:20,'LineWidth',2)
xlabel('\lambda_0')
ylabel('\lambda_{max}')
legend({'scatter','\lambda_0=\lambda_{max}'})

figure(2)
hold on
scatter(A(3,:),A(4,:))
plot(0.4:0.1:1,0.4:0.1:1,'LineWidth',2)
xlabel('|uv|^2_{sim}')
ylabel('|uv|^2_{eq}')
legend({'scatter','|uv|^2_{sim}=|uv|^2_{eq}'})



function [lam_real,lam_es,uv_real,uv_es] = RM(p,n,lam0)
gamma = p/n;
R = lam0;
u = randn(p,1);
u = u / sqrt(u'*u);
%u = eye(p,1);
Sigma = eye(p) + lam0*u*u';
x = mvnrnd(zeros(p,1),Sigma,n)';

Sn = 1/n*x*x';

[V, D] = eig(Sn);
% Extract the eigenvalues and eigenvectors
eigenvalues = diag(D);
eigenvectors = V';

% Sort the eigenvalues in descending order
[sorted_eigenvalues, sort_index] = sort(eigenvalues, 'descend');
sorted_eigenvectors = eigenvectors(sort_index, :);

lam_real = max(sorted_eigenvalues);
lam_es = (1+R)*(1+gamma/R);
v_max = sorted_eigenvectors(1,:)';

uv_real = (u'*v_max)^2;
uv_es = (1 - gamma/R^2)/(1+gamma/R);
end

