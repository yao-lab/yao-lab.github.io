function [W,Z] = randomForest(X, X_ko, y, ntrees, plotTF)
% KNOCKOFFS.STATS.RANDOMFOREST  The random forest feature importance difference W
%   [W, Z] = KNOCKOFFS.STATS.RANDOMFOREST(X, X_ko, y)
%   [W, Z] = KNOCKOFFS.STATS.RANDOMFOREST(X, X_ko, y, ntrees)
%   [W, Z] = KNOCKOFFS.STATS.RANDOMFOREST(X, X_ko, y, ntrees, plot)
%
%   Computes the statistic
%
%     W_j = |Z_j| - |\tilde Z_j|,
%
%   where Z_j and \tilde Z_j are the features importances of the 
%   jth variable and its knockoff, respectively, resulting from
%   fitting a random forest.
%
%   The importance of a variable is measured as the total decrease
%   in node impurities from splitting on that variable, averaged over all trees. 
%   For regression, the node impurity is measured by residual sum of squares.
%   For classification, it is measured by the Gini index.

if ~exist('ntrees', 'var'), ntrees = 1000; end
if ~exist('plotTF', 'var'), plotTF = false; end

p = size(X,2);

%TODO: maybe parallelize this
B = TreeBagger(ntrees,[X X_ko],y,'method','regression','OOBPredictorImportance','On');%,'NumPrint',10);
Z = B.OOBPermutedPredictorDeltaError;
if plotTF, plot(oobError(B)); end
W = abs(Z(1:p))-abs(Z((p+1):(2*p)));

end