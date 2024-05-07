% (a)
X = importdata('train.6.txt');
X = X';
% (b)
sample_mean = mean(X);
X_tilde = X - sample_mean;
% (c)
[U,S,V] = svd(X_tilde);
% (d)
covariance_matrix = X_tilde * X_tilde'/664;
e = eig(covariance_matrix);
e = sort(e,'descend');
plot(e(1:10)/sum(e))
% (e)
imshow(mean(U,2))
imshow(U(:,1:10))
% (f)
v1 = V(1,:);
[~,order] = sort(v1, 'ascend');
% (g)
v2 = V(2,:);
scatter(v1,v2,'.b')