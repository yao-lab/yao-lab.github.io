clear all;
clc;
load('snp452-data.mat')
%% Separate dataset
[m,p] = size(X);    

% R
R = zeros(m-1,p);
for t = 1:m-1
    R(t,:) = (X(t+1,:)-X(t,:))./X(t,:);   % dot division
end

% histroy
T = floor((m-2)/2);
History = R(1:T,:);
Test = R(T+1:end,:);
% prediction


%% Benchmark 1: EWP
weight_EWP = (1/p)*ones(p,1);
return_EWP = Test*weight_EWP;
sharpe_EWP = mean(return_EWP)/std(return_EWP)*sqrt(252);
mean_EWP = mean(return_EWP)*252;
std_EWP = std(return_EWP)*sqrt(252);
values_EWP = cumprod(1+return_EWP);


%% Benchmark 2: MV
% sample mean 
mean_MLE = mean(History,1);

% JSE 
% var = var(History,1);
% mean_JSE = (1-(p-2)*var/norm(mean_MLE,2)^2).*mean_MLE;
% centered sample cov   

cov_MLE = cov(History-mean_MLE);
[weight_MV,sharpe_MV,values_MV,mean_MV,std_MV] = performance(Test,mean_MLE,cov_MLE,1);


%% Benchmark 3: GMV
PI = cov_MLE^-1;
weight_GMV = PI*ones(p,1)/(ones(1,p)*PI*ones(p,1));
weight_GMV = weight_GMV/sum(abs(weight_GMV)); % normalize
return_GMV = Test*weight_GMV;
sharpe_GMV = mean(return_GMV)/std(return_GMV)*sqrt(252);
mean_GMV = mean(return_GMV)*252;
std_GMV = std(return_GMV)*sqrt(252);
values_GMV = cumprod(1+return_GMV);

%% shrinkage estimator 1

n = T;
S = cov_MLE;
P = (n-p-2)/(n-1)*S^(-1);

alpha = (n-p-2/p*(n-p-2))/(n-p-1);
beta = (n-p-2-2/p)/(n-p-1);
rho = (alpha*trace(P^2)+beta*trace(P)^2)/((alpha+n-p-4)*(trace(P^2)-trace(P)^2/p));
rho = min(rho,1);
PI_JS1 = rho*trace(P)/p*eye(p)+(1-rho)*P;

[weight_JS1,sharpe_JS1,values_JS1,mean_JS1,std_JS1] = performance(Test,mean_MLE,PI_JS1,2);

%% shrinkage estimator 2
diagP = diag(diag(P));
numerator = 2*trace(diagP^2)+(n-p)/(n-p-1)*trace(P^2)+(n-p-2)/(n-p-1)*(trace(P)^2);
denominator = ((n-p)/(n-p-1)+n-p-4)*(trace(P^2)-trace(diagP^2));
rho = min(numerator/denominator,1);
PI_JS2 = rho*diagP+(1-rho)*P;


[weight_JS2,sharpe_JS2,values_JS2,mean_JS2,std_JS2] = performance(Test,mean_MLE,PI_JS2,2);

%% OAS estimator for covariance matrix
S = cov_MLE;

numerator = (1-2/p)*trace(S^2)+trace(S)^2;
denominator = (n-2/p)*(trace(S^2)-trace(S)^2/p);
rho = min(numerator/denominator,1);
S_OAS1 = rho*trace(S)/p*eye(p)+(1-rho)*S;

[weight_OAS1,sharpe_OAS1,values_OAS1,mean_OAS1,std_OAS1] = performance(Test,mean_MLE,S_OAS1,1);
%% OAS estimator 2 for covariance matrix
diagS = diag(diag(S));
numerator = trace(S^2)+trace(S)^2-2*trace(diagS^2);
denominator = n*(trace(S^2)-trace(diagS^2));
rho = min(numerator/denominator,1);
S_OAS2 = rho*diagS+(1-rho)*S;

[weight_OAS2,sharpe_OAS2,values_OAS2,mean_OAS2,std_OAS2] = performance(Test,mean_MLE,S_OAS2,1);


%% test
%% Compare the performance
figure
plot(1:m-T-1,values_EWP)
hold on
plot(1:m-T-1,values_MV)
plot(1:m-T-1,values_GMV)
plot(1:m-T-1,values_JS1)
plot(1:m-T-1,values_JS2)
plot(1:m-T-1,values_OAS1)
plot(1:m-T-1,values_OAS2)
legend(["EWP","MV","GMV","PM1","PM2","CM1","CM2"])
xlabel("Time")
ylabel("Portfolio Returns")
hold off

figure
idx = [1:100:452];
plot(X(:,idx))
legend("Stock 1","Stock 101","Stock 201","Stock 301","Stock 401")
xlabel("Time")
ylabel("Prices")


