% Split Knockoffs, the data addaptive filter for controlling False
% Discovery Rate (FDR) in structural sparsity setting.
%
% Reference: https://arxiv.org/abs/2103.16159
% 
% Files
%   create    : Create the linear regression model for split LASSO as well
%       as the Split Knockoff copy from given data. 
%   filter    : Run the Split Knockoff filter on the data set.
%   cv_filter : Run the Split Knockoff filter on the data set with cross
%       validation choice of nu.
%
% For examples on simulation experiments, see the scripts in "simulation"
%   dictionary.
% For examples on Alzheimer’s Disease, see the scripts in "AD_experiments"
%   dictionary.