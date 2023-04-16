clear all;
clc;
load('snp452-data.mat')
[m,p] = size(X);    

% R
R = zeros(m-1,p);
for t = 1:m-1
    R(t,:) = (X(t+1,:)-X(t,:))./X(t,:);   % dot division
end
mu = mean(R,1);
Sigma = cov(R);
invSigma = Sigma^-1;

n = 600;
nTrials = 500;
errors = zeros(nTrials,3);
alpha = (n-p-2/p*(n-p-2))/(n-p-1);
beta = (n-p-2-2/p)/(n-p-1);
for i = 1:nTrials
    Y = mvnrnd(zeros(1,p),Sigma,n);   %
    
    % MLE
    cov_MLE = cov(Y);   % sample covariance matrix
    error_MLE = norm(invSigma-cov_MLE^-1,"fro")/norm(invSigma,"fro");

    S = cov_MLE;
    P = (n-p-2)/(n-1)*S^(-1);
    % PM1
    rho = (alpha*trace(P^2)+beta*trace(P)^2)/((alpha+n-p-4)*(trace(P^2)-trace(P)^2/p));
    rho = min(rho,1);
    PI_JS1 = rho*trace(P)/p*eye(p)+(1-rho)*P;
    error_JS1 = norm(invSigma-PI_JS1,"fro")/norm(invSigma,"fro");
 
    % PM2
    diagP = diag(diag(P));
    numerator = 2*trace(diagP^2)+(n-p)/(n-p-1)*trace(P^2)+(n-p-2)/(n-p-1)*(trace(P)^2);
    denominator = ((n-p)/(n-p-1)+n-p-4)*(trace(P^2)-trace(diagP^2));
    rho = min(numerator/denominator,1);
    PI_JS2 = rho*diagP+(1-rho)*P;
    error_JS2 = norm(invSigma-PI_JS2,"fro")/norm(invSigma,"fro");
    
    errors(i,1) = error_MLE;
    errors(i,2) = error_JS1; 
    errors(i,3) = error_JS2; 
    
end

figure
boxchart(errors)
ylabel("Relative Error")
%set(gca,'Yscale','log')
set(gca,'TickLabelInterpreter','latex');
set(gca,'XTickLabel',{'$$\widehat{\Pi}^{(0)}$$','$$\widehat{\Pi}^{(1)}$$','$$\widehat{\Pi}^{(2)}$$'});

figure
boxchart(errors(:,2:3))
ylabel("Relative Error")
%set(gca,'Yscale','log')
set(gca,'TickLabelInterpreter','latex');
set(gca,'XTickLabel',{'$$\widehat{\Pi}^{(1)}$$','$$\widehat{\Pi}^{(2)}$$'});