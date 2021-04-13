function [W,Z] = stabilitySignedMax_bin(X, X_ko, y, weakness, nRep)
% KNOCKOFFS.STATS.STABILITYSIGNEDMAX_BIN  Signed difference of stability 
% selection W (binomial response)
%   [W, Z] = KNOCKOFFS.STATS.STABILITYSIGNEDMAX_BIN(X, X_ko, y)
%   [W, Z] = KNOCKOFFS.STATS.STABILITYSIGNEDMAX_BIN(X, X_ko, y, weakness)
%   [W, Z] = KNOCKOFFS.STATS.STABILITYSIGNEDMAX_BIN(X, X_ko, y, weakness, nRep)
%
%   Computes the statistic
%
%     W_j = max(Z_j, \tilde Z_j) * sgn(Z_j - \tilde Z_j),
%
%   where Z_j and \tilde Z_j are the stability selection probabilities
%   values of the jth variable and its knockoff, respectively, resulting 
%   from repeated randomized logistic regression with L1 regularization.
%
% See also KNOCKOFFS.STATS.STABILITYSIGNEDMAX

if ~exist('weakness', 'var'), weakness = 0.2; end
if ~exist('nRep', 'var'), nRep = 100; end
tol = 1e-6;

[n,p] = size(X);

options = glmnetSet();
options.standardize = true;
options.intr = true;
options.standardize_resp = false;
options.alpha = 1; % lasso regression

nsub = floor(n/2);

fit = glmnet(bsxfun(@times,[X(1:nsub,:) X_ko(1:nsub,:)],rand(1,2*p)*(1-weakness)+weakness), y(1:nsub), 'binomial', options);
options.lambda = fit.lambda;
probs = zeros(2*p,length(options.lambda));

for i = 1:nRep
    indsub = randperm(n,nsub);
    Xsub = bsxfun(@times,[X(indsub,:) X_ko(indsub,:)],rand(1,2*p)*(1-weakness)+weakness);
    fit = glmnet(Xsub,y(indsub),'binomial',options);
    probs = probs + (abs(fit.beta)>tol)/nRep;
end
Z = max(probs,[],2);
orig = 1:p; ko = (p+1):(2*p);
W = max(Z(orig), Z(ko)) .* sign(Z(orig) - Z(ko));

end