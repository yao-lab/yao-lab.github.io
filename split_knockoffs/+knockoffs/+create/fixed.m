function X_k = fixed(X, method, randomize)
%KNOCKOFFS.CREATE.FIXED Creates fixed-X knockoffs.
%
%  X_k = KNOCKOFFS.CREATE.FIXED(X)
%  X_k = KNOCKOFFS.CREATE.FIXED(X, method)
%  X_k = KNOCKOFFS.CREATE.FIXED(X, method, randomize)
%
%  Inputs:
%       X    - n x p scaled covariate matrix
%     method - either 'equi' (for equi-correlated knockoffs), 'sdp'
%              (for knockoffs optimized using semi-definite programming) or 
%              'asdp' (for approximate SDP knockoffs)
%              Default: 'sdp'
%  randomize - whether to use randomization in the construction of the
%              knockoff variables.
%   
%  Outputs:
%       X_k - n x p matrix of knockoff variables
%
%  See also KNOCKOFFS.CREATE.GAUSSIAN

if ~exist('method', 'var') || isempty(method), method = 'sdp'; end;
if ~exist('randomize', 'var'), randomize = []; end;

% Create the knockoffs
method = lower(method);
switch method
    case 'equi'
        X_k = knockoffs.create.fixed_equi(X, randomize);
    case 'sdp'
        X_k = knockoffs.create.fixed_SDP(X, randomize, false);
    case 'asdp'
        X_k = knockoffs.create.fixed_SDP(X, randomize, true);
    case 'mincondcov'
        X_k = knockoffs.create.fixed_MinCondCov(X, randomize);
    otherwise
        error('Invalid Fixed-X knockoff creation method %s', method)
end