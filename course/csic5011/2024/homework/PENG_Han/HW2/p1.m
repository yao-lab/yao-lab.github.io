clear
clc

load('snp452-data.mat')
% a
Y = log(X);
Y = Y';

% b
dY = Y(:,2:end);
for i=1:1257
    dY(:,i) = Y(:,i+1) - Y(:,i);
end

% c
COV =  dY*dY'/1257;

% d
% Compute the eigenvalues and eigenvectors
[V, D] = eig(COV);

% Extract the eigenvalues and eigenvectors
eigenvalues = diag(D);
eigenvectors = V';

% Sort the eigenvalues in descending order
[sorted_eigenvalues, sort_index] = sort(eigenvalues, 'descend');
sorted_eigenvectors = eigenvectors(sort_index, :);

% Store the sorted eigenvalues and eigenvectors
lambda_hat = sorted_eigenvalues;
eigenvectors_hat = sorted_eigenvectors';
figure(1)
plot(lambda_hat(1:10))
xlabel('i')
ylabel('\lambda')

% e
R  = 100;
LAM = zeros(100,452);
y = dY;
for k = 1:R 
    for i=2:452
        PI = randperm(1257);
        y(i,:) = dY(i,PI);
    end
    COV_y =  y*y'/1257;
    [~, Dy] = eig(COV_y);
    [sorted_lam, ~] = sort(diag(Dy), 'descend');
    LAM(k,:) = sorted_lam;
end

pval = zeros(R,452);
for i=1:452
    for k = 1:R 
        if sorted_eigenvalues(i)>LAM(k,i)
            pval(k,i) = 1;
        end
    end
end
pvalSums = (sum(pval, 1)+1)/(R+1);
index = find(pvalSums>0.95, 1, 'last' );
figure(2)
loglog(lambda_hat,'LineWidth',2)
hold on
loglog(LAM(1,:),'LineWidth',2)
loglog(index*ones(452,1),lambda_hat(1:452),'--','LineWidth',2)
legend({'original', 'permuted top 5%', 'truncated position'})
xticks([index, 452]) 
xlabel('dimension (log scale)')
ylabel('\lambda')


