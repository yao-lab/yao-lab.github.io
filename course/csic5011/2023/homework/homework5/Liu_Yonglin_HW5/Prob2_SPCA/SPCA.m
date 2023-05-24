N = 1000;p=10;
X = [];

%Compute the sample covariance matric with n=1000 samples
for j = 1:1000
    x = [];
    for i=1:4
        x(end+1)=290.*randn+randn;
    end
    for i=5:8
        x(end+1)=300.*randn+randn;
    end
    for i=9:10
        V1 = 290.*randn;
        V2 = 300.*randn;
        x(end+1)=-0.3*V1+0.925*V2+randn;
    end
    X =[X;x];
end

X = X';
S = 1/N * (X*X');

% Compute the top 4 principal component of S using eigenvector
% decomposition
[U,B,V]=svds(S, 4);

e = ones(p, 1);
%lambda = 5;

% Compare result of cvx and normal PCA
for lambda=1:10
    [U,B,V]=svds(S, 2)
    cvx_begin
        variable X(p, p) symmetric;
        X == semidefinite(p);
        minimize(-trace(S*X)+lambda*(e'*abs(X)*e));
        subject to 
            trace(X) == 1;
    cvx_end
    % Show the first two principal components
    figure(lambda);
    subplot(2, 1, 1);
    scatter(X(:, 1), X(:, 2),'o');
    subplot(2, 1, 2);
    scatter(U(:, 1), U(:, 2),'o');
    
    disp('0-norm of Sparse PCAs biggest eigenvalue is:')
    sum(abs(X(:, 1))>0)
    disp('0-norm of PCAs biggest eigenvalue is:')
    sum(abs(U(:, 1))>0)
end

% Do the 1-4 the Sparse PCA and normal PCA
for i=1:4
    [U,B,V]=svds(S, 2)
    cvx_begin
        variable X(p, p) symmetric;
        X == semidefinite(p);
        minimize(-trace(S*X)+lambda*(e'*abs(X)*e));
        subject to 
            trace(X) == 1;
    cvx_end
    % Show the first two principal components
    figure(i);
    subplot(2, 1, 1);
    scatter(X(:, 1), X(:, 2),'o');
    title('Plot of first two components of Sparse PCA');
    subplot(2, 1, 2);
    scatter(U(:, 1), U(:, 2),'o');
    title('Plot of first two components of Normal PCA');
    
    disp('0-norm of Sparse PCAs biggest eigenvalue is:')
    sum(abs(X(:, 1))>0)
    disp('0-norm of PCAs biggest eigenvalue is:')
    sum(abs(U(:, 1))>0)
    S = S-X;
end

