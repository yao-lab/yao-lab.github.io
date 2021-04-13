function Y = normc(X)
%KNOCKOFFS.PRIVATE.NORMC Normalize columns of a matrix.
%   A clone of NORMC from the Neural Network toolbox.

n = size(X,1);
X = bsxfun(@minus,X,mean(X,1));
factors = 1 ./ sqrt(sum(X.^2, 1));
Y = X .* factors(ones(1,n),:);

end