X=reshape(Y,[size(Y,1)*size(Y,2) size(Y,3)]);
XX=X';
proximities = zeros(size(XX,1));
for i=1:size(XX,1)
    for j =1:size(XX,1)
        proximities(i,j) = pdist2(XX(i,:),XX(j,:),'euclidean');
    end
end
n = size(proximities,1);
identity = eye(n);
one = ones(n);
centering_matrix = identity - (1/n) * one;
J = centering_matrix;
B = -.5*J*(proximities).*(proximities)*J;
M = 10; 
[eigvec,eigval] = eig(B);
[eigval, order] = sort(max(eigval)','descend');
eigvec = eigvec(order,:);
eigvec = eigvec(:,1:M); 
eigval = eigval(1:M);
A = zeros(10);
A(1:11:end) = eigval;
X = eigvec*A;
plot(eigval);
figure (2)
plot(X(:,1),X(:,2),'o')
title('2-D MDS embedding');
gname;
