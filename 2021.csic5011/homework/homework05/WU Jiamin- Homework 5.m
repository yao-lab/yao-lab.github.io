%Question1
m=20;
n=20;
p=0.1;
r=1;
A = randn(20,20);
[U,S,V] = svds(A,3);
L0 = U(:,1:r)*V(:,1:r)';
E0 = rand(20);
E = 1*abs(E0>1-p);
X = L0 + E;
lambda = 0.25;
cvx_begin
    variable L(20,20);
    variable S(20,20);
    variable W1(20,20);
    variable W2(20,20);
    variable Y(40,40) symmetric;
    Y == semidefinite(40);
    minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S))));
    subject to
        L + S >= X-1e-5;
        L + S <= X + 1e-5;
        Y == [W1, L';L W2];
cvx_end
disp('$\|S-S0\|_infty$:')
norm(S-E,'inf')
disp('$\|L-L0\|$:')
norm(L-L0)
suc_p=[]
for i= 0.1:0.2:0.9
    E1 = 1*abs(E0>1-i);
    X1 = L0 + E1;
    lambda = 0.25;
    cvx_begin
        variable L(20,20);
        variable S(20,20);
        variable W1(20,20);
        variable W2(20,20);
        variable Y(40,40) symmetric;
        Y == semidefinite(40);
        minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S))));
        subject to
            L + S >= X1-1e-5; 
            L + S <= X1 + 1e-5;
            Y == [W1, L';L W2];
    cvx_end
    mu_S0=norm(S)
    ksi_L0=norm(L,'inf')
if mu_S0*ksi_L0<1/6
        fprintf('success for p= : %d feet.\n', i);
        disp('p=1');      
        suc_p=[suc_p,i];
    else
        disp('p=0');
    end
end

L1=U(:,2)*V(:,2)';
X2 = L1 + E;
lambda = 0.25;
cvx_begin
    variable L(20,20);
    variable S(20,20);
    variable W1(20,20);
    variable W2(20,20);
    variable Y(40,40) symmetric;
    Y == semidefinite(40);
    minimize(.5*trace(W1)+0.5*trace(W2)+lambda*sum(sum(abs(S))));
    subject to
        L + S >= X2-1e-5;
        L + S <= X2 + 1e-5;
        Y == [W1, L';L W2];
cvx_end
mu_S0=norm(S)
ksi_L0=norm(L,'inf')
if mu_S0*ksi_L0<1/6
    fprintf('success for r= : %d feet.\n', 2);
    disp('p=1');
else
    disp('p=0');
end
L = randn(m,r) * randn(r,n);    
S = sprandn(m,n,0.05);
S(S ~= 0) = 20*binornd(1,0.5,nnz(S),1)-10;
V = 0.01*randn(m,n); 
A = S + L + V;
g2_max = norm(A(:),inf);
g3_max = norm(A);
g2 = 0.15*g2_max;
g3 = 0.15*g3_max;
m=1000;
n=1000;
N=3;
e=1e-4;
ep=1e-2;
tic;
lambda = 1;
rho = 1/lambda;
X_1 = zeros(m,n);
X_2 = zeros(m,n);
X_3 = zeros(m,n);
z   = zeros(m,N*n);
U   = zeros(m,n);
for k = 1:10
    B = avg(X_1, X_2, X_3) - A./N + U;
    X_1 = (1/(1+lambda))*(X_1 - B);
    X_2 = prox_l1(X_2 - B, lambda*g2);
    X_3 = prox_matrix(X_3 - B, lambda*g3, @prox_l1);
    x = [X_1 X_2 X_3];
    zold = z;
    z = x + repmat(-avg(X_1, X_2, X_3) + A./N, 1, N);
    U = B;
    h.objval(k)   = objective(X_1, g2, X_2, g3, X_3);
    h.r_norm(k)   = norm(x - z,'fro');
    h.s_norm(k)   = norm(-rho*(z - zold),'fro');
    h.eps_pri(k)  = sqrt(m*n*N)*ABSTOL + RELTOL*max(norm(x,'fro'), norm(-z,'fro'));
    h.eps_dual(k) = sqrt(m*n*N)*ABSTOL + RELTOL*sqrt(N)*norm(rho*U,'fro');
    if h.r_norm(k) < h.eps_pri(k) && h.s_norm(k) < h.eps_dual(k)
         break;
    end

end

Question 2
n=1000;
V1=normrnd(0, 290);
V2=normrnd(0, 200);
V3=-0.3*V1+0.925*V2+normrnd(0,1);
X0=zeros(10,n);
for i=1:1:4
    X0(i,:)=V1*ones(1,n)+randn(1,n);
end
for i=5:1:8
    X0(i,:)=V2*ones(1,n)+randn(1,n);
end
for i=9:1:10
    X0(i,:)=V3*ones(1,n)+randn(1,n);
end
Sigma=cov(X0');
[eig_vec, eig_val]=eig(Sigma,'vector');
[eig_val, ind] = sort(eig_val, 'descend');
eig_vec = eig_vec(:, ind);
top4_pca=eig_vec(1:4,:);

R = X0*X0';
d = 10;
e = ones(d,1);
for lambda = 0:2:6

    cvx_begin
        variable X(d,d) symmetric;
        X == semidefinite(d);
        minimize(-trace(R*X)+lambda*(e'*abs(X)*e));
        subject to
            trace(X)==1;
    cvx_end
    top1_spca=X(:,1);
    disp('$\|pca-spca\|$ for lambda = ');
    disp(lambda);
    disp(norm(top1_spca-top4_pca(1,:)));
end

lambda = 5;
cvx_begin
    variable X(d,d) symmetric;
    X == semidefinite(d);
    minimize(-trace(R*X)+lambda*(e'*abs(X)*e));
    subject to
        trace(X)==1;
cvx_end
top1_spca=X(:,1);

R1=R-X;
cvx_begin
    variable X1(d,d) symmetric;
    X1 == semidefinite(d);
    minimize(-trace(R1*X1)+lambda*(e'*abs(X1)*e));
    subject to
        trace(X1)==1;
cvx_end
top2_spca=X1(:,1);

disp('$\|pca-spca\|$');
disp(norm(top2_spca-top4_pca(2,:)));

R2=R1-X1;
cvx_begin
    variable X2(d,d) symmetric;
    X2 == semidefinite(d);
    minimize(-trace(R2*X2)+lambda*(e'*abs(X2)*e));
    subject to
        trace(X2)==1;
cvx_end
top3_spca=X2(:,1);

disp('$\|pca-spca\|$');
disp(norm(top3_spca-top4_pca(3,:)));

R3=R2-X2;
cvx_begin
    variable X3(d,d) symmetric;
    X3 == semidefinite(d);
    minimize(-trace(R3*X3)+lambda*(e'*abs(X3)*e));
    subject to
        trace(X3)==1;
cvx_end
top4_spca=X3(:,1);

disp('$\|pca-spca\|$');
disp(norm(top4_spca-top4_pca(4,:)));

