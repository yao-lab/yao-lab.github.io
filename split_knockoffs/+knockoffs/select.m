function S = select(W, q, method)
% KNOCKOFFS.SELECT  Select variables based on the knockoff statistics
%
%   S = KNOCKOFFS.SELECT(W, q) select using 'knockoff' method
%   S = KNOCKOFFS.SELECT(W, q, method) select with given method
%
%   Inputs:
%       W - statistics W_j for testing null hypothesis beta_j = 0.
%       q - target FDR
%       method - either 'knockoff' or 'knockoff+'
%                Default: 'knockoff+'
%
%   Outputs:
%       S - array of selected variable indices
%
%   See also KNOCKOFFS.THRESHOLD.

if ~exist('method', 'var')
    method = [];
end

W = reshape(W, 1, []);
T = knockoffs.threshold(W, q, method);
S = find(W >= T);

end