V9=290*0.09+300*0.925*0.925+2;
V9V5=0.925*300;
V9V1=-0.3*290;
True_sigma=[
    291 , 290 , 290 , 290 , 0   , 0  , 0  , 0   , V9V1 , V9V1 ,
    290 , 291 , 290 , 290 , 0   , 0  , 0  , 0   , V9V1 , V9V1 ,
    290 , 290 , 291 , 290 , 0   , 0  , 0  , 0   , V9V1 , V9V1 ,
    290 , 290 , 290 , 291 , 0   , 0  , 0  , 0   , V9V1 , V9V1 ,
    0   , 0   , 0   , 0   , 301 , 300, 300, 300 , V9V5 , V9V5 ,
    0   , 0   , 0   , 0   , 300 , 301, 300, 300 , V9V5 , V9V5 ,
    0   , 0   , 0   , 0   , 300 , 300, 301, 300 , V9V5 , V9V5 ,
    0   , 0   , 0   , 0   , 300 , 300, 300, 301 , V9V5 , V9V5 ,
    V9V1, V9V1, V9V1, V9V1, V9V5,V9V5,V9V5, V9V5,V9    , V9-1 ,
    V9V1, V9V1, V9V1, V9V1, V9V5,V9V5,V9V5, V9V5,V9-1  , V9   ,
];
[V,D] = eig(True_sigma);
[d,ind] = sort(diag(D));
D = D(ind,ind);
V = V(:,ind);
True_sigma_rank4=V(1:10,1:4)*V(1:10,1:4)';
norm(True_sigma_rank4-True_sigma);

data=gen_data(1000);
sigma=cov(data);
norm(sigma-True_sigma,"fro");

% a,b,c=eig(sigma);
sigma_S_prox=0
lambda=0.1;

S_pca_1=sPCA_(sigma,lambda);
sigma_new=sigma-trace(sigma*S_pca_1)*S_pca_1;
sigma_S_prox=sigma_S_prox+trace(sigma*S_pca_1)*S_pca_1;
disp("here")
norm(sigma_S_prox-True_sigma,"inf")

S_pca_2=sPCA_(sigma_new,lambda);
sigma_S_prox=sigma_S_prox+trace(sigma_new*S_pca_2)*S_pca_2;
sigma_new=sigma_new-trace(sigma_new*S_pca_2)*S_pca_2;
disp("here")
norm(sigma_S_prox-True_sigma,"inf")

S_pca_3=sPCA_(sigma_new,lambda);
sigma_S_prox=sigma_S_prox+trace(sigma_new*S_pca_3)*S_pca_3;
sigma_new=sigma_new-trace(sigma_new*S_pca_3)*S_pca_3;
disp("here")
norm(sigma_S_prox-True_sigma,"inf")

S_pca_4=sPCA_(sigma_new,lambda);
sigma_S_prox=sigma_S_prox+trace(sigma_new*S_pca_4)*S_pca_4;
disp("here")
norm(sigma_S_prox-True_sigma,"inf")

% S_pca_vec=[S_pca_1,S_pca_2,S_pca_3,S_pca_4];
% sigma_new=sigma_new-trace(sigma_new*S_pca_4)*S_pca_4;

% S_pca_1=sPCA_(sigma,lambda);
% S_pca_1=sPCA_(sigma,lambda);

[u,s,v]=svds(sigma,4);
pca_rank1=u(:,1:1)*v(:,1:1)';
pca_rank2=u(:,2)*v(:,2)';
pca_rank3=u(:,3)*v(:,3)';
pca_rank4=u(:,4)*v(:,4)';

norm(pca_rank1-S_pca_1,'inf')
norm(pca_rank2-S_pca_2,'inf')
norm(pca_rank3-S_pca_3,'inf')
norm(pca_rank4-S_pca_4,'inf')

norm(sigma_S_prox-True_sigma,'fro')

function X=sPCA_(sigma,lambda)
R = sigma;
d = 10;
e = ones(d,1);
cvx_begin
variable X(d,d) symmetric;
X == semidefinite(d); 
minimize(-trace(R*X)+lambda*(e'*abs(X)*e)); 
subject to 
    trace(X)==1; 
cvx_end
end

function data=gen_data(n)
data=[];
for i=1:n
    V_1=randn(1)*sqrt(290);
    V_2=randn(1)*sqrt(300);
    V_3=-0.3*V_1+0.925*V_2+randn(1);
    X=[V_1+randn(1),V_1+randn(1),V_1+randn(1),V_1+randn(1),V_2+randn(1),V_2+randn(1),V_2+randn(1),V_2+randn(1),V_3+randn(1),V_3+randn(1)];
    data=[data;X];
end
end




