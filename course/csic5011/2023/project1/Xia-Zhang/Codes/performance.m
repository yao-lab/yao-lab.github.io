function [weights,sharpe,values,mean_hat,std_hat] = performance(Test,mu,PI,flag)
%performance Summary of this function goes here
%   
%  input
%  Test -- t*p matrix: test data 
%  mu   -- 1*p vector: estimation for mean
%  flag = 1
%  S    -- p*p matrix: estimation for the Covariance Matrix
%  flag = 2
%  S    -- p*p matrix: estimation for the Precision Matrix
%%%

p = length(mu);
weights = zeros(p,1);
% compute weights
if flag == 1
    weights = PI^(-1)*mu';
elseif flag == 2
    weights = PI*mu';
end
% normalize
weights = weights/sum(abs(weights)); 
returns = Test*weights;
sharpe = mean(returns)/std(returns)*sqrt(252);
mean_hat = mean(returns)*252;
std_hat = std(returns)*sqrt(252);
values = cumprod(1+returns);
end

