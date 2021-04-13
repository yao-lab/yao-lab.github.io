function Y = normc(X)
% Normalize columns of a matrix.
%   A clone of NORMC from the Neural Network toolbox.

n = size(X,1);
X = bsxfun(@minus,X,mean(X,1));
factors = 1 ./ sqrt(sum(X.^2, 1));
% Y = X .* factors(ones(1,n),:);    This is used in Knockoffs.
Y = X .* factors(ones(1,n),:) * sqrt(n-1); 

end