function T = threshold(W, q, method)
% KNOCKOFFS.THRESHOLD  Compute the threshold for variable selection
%   T = KNOCKOFFS.THRESHOLD(W, q) threshold using 'knockoff' method
%   T = KNOCKOFFS.THRESHOLD(W, q, method) threshold with given method
%
%   Inputs:
%       W - statistics W_j for testing null hypothesis beta_j = 0.
%       q - target FDR
%       method - either 'knockoff' or 'knockoff+'
%                Default: 'knockoff'
%
%   Outputs:
%       T - threshold for variable selection
%
%   See also KNOCKOFFS.SELECT.

if ~exist('method', 'var') || isempty(method)
    method = 'knockoff+';
end

switch lower(method)
    case 'knockoff'
        offset = 0;
    case 'knockoff+'
        offset = 1;
    otherwise
        error('Invalid threshold method %s', method)
end

W = reshape(W, 1, []);
t = sort([0 abs(W(W~=0))]);
ratio = zeros(1, length(t));
for i = 1:length(t)
    ratio(i) = (offset + sum(W <= -t(i))) / max(1, sum(W >= t(i)));
end

index = find(ratio <= q, 1, 'first');
if isempty(index)
    T = Inf;
else
    T = t(index);
end

end