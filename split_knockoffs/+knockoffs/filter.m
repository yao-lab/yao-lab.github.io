function [S, W, Z] = filter(X, y, q, Xmodel, varargin)
% KNOCKOFFS.FILTER  Run the knockoff filter on a data set.
%
%   [S, W] = KNOCKOFFS.FILTER(X, y, q, Xmodel, ...)
%
%   This function runs the knockoff procedure from start to finish,
%   creating the knockoffs, computing the test statistics, and selecting
%   variables. It is the main entry point for the knockoff package.
%
%   Inputs:
%       X   - n x p scaled predictor matrix or table (n > p)
%       y   - n x 1 response vector
%       q   - Target false discovery rate (FDR)
%    Xmodel - A cell array describing the model for the covariates
%             Allowed values (see KNOCKOFFS.CREATE):
%                 - {'gaussian', mu, Sigma}  (Model-X Gaussian variables, model-free response)
%                 - {'fixed'}                (Fixed design, assumes linear regression response)  
%             
%   Optional Inputs:
%       'Statistics' - A handle to a function f(X, X_knockoffs, y) that
%                      computes the test statistics. By default, the lasso
%                      coefficient difference with cross-validation
%                      (LCD) is used with 'Model-X' variables and the
%                      lassoLambdaSignedMax statistics are used with 'Fixed-X' variables.
%       'Method'     - Method to use for creating knockoffs. The available 
%                      options depend on 'Xmodel'. See also KNOCKOFFS.CREATE.
%       'Threshold'  - Method to use for thresholding (default: 'knockoff+').
%                      See also KNOCKOFFS.SELECT.
%       'Randomize'  - (Only for 'fixed' model)  Whether to use 
%                      randomization in the construction of knockoffs and 
%                      (when n<2p) augmenting the model with extra rows.
%                      Default: no.
%
%   Ouputs:
%       S - Column vector of selected variables. Contains indices if X is
%           a matrix and variable names if X is a table.
%       W - 1 x 2p vector of test statistics computed for the variables and
%           their knockoffs
%
%   See also KNOCKOFFS.CREATE, KNOCKOFFS.SELECT.

preparser = inputParser;
preparser.CaseSensitive = false;
if (~verLessThan('matlab', '8.2')) % R2013b or later
    preparser.PartialMatching = false;
end

istable_safe = @(x) ~verLessThan('matlab', '8.2') && istable(x);
preparser.addRequired('X', @(x) isnumeric(x) || istable_safe(x));
preparser.addRequired('y', @isnumeric);
preparser.addRequired('q', @(x) isnumeric(x) && isscalar(x));
preparser.addRequired('Xmodel', @(x) iscell(x));
parser = preparser;
preparser.parse(X, y, q, Xmodel);

if isequal(Xmodel{1}, 'fixed')
  parser.addOptional('Statistics', ...
    @knockoffs.stats.lassoLambdaSignedMax, ...
    @(x) isa(x, 'function_handle'));
else
  parser.addOptional('Statistics', ...
    @knockoffs.stats.lassoCoefDiff, ...
    @(x) isa(x, 'function_handle'));
end

parser.addOptional('Method', []);
parser.addOptional('Threshold', []);
parser.addOptional('Randomize', false, @islogical);
parser.parse(X, y, q, Xmodel, varargin{:});

% Extract variable names.
if (istable_safe(X))
    Xnames = X.Properties.VariableNames;
    X = table2array(X);
else
    Xnames = {};
end

% Verify dimensions (only for 'fixed' knockoffs)
if isequal(Xmodel{1}, 'fixed')
    [n,p] = size(X);
    if (n <= p)
        error('knockoff:DimensionError', 'Data matrix for this model must have n > p')
    elseif (n < 2*p)
        warning('knockoff:DimensionWarning', ...
            'Data matrix has p < n < 2*p. Augmenting the model with extra rows.');
        [U,~,~] = svd(X);
        U_2 = U(:,(p+1):n);
        sigma = sqrt(mean((U_2'*y).^2)); % = sqrt(RSS/(n-p))
        if (parser.Results.Randomize)
            y_extra = randn(2*p-n,1)*sigma;
        else
            seed = rng(0);
            y_extra = randn(2*p-n,1)*sigma;
            rng(seed);
        end
        y = [y;y_extra];
        X = [X;zeros(2*p-n,p)];
    end
    
    % Normalize columns of X
    X = knockoffs.private.normc(X);
end

% Run the knockoff filter.
X_k = knockoffs.create(X, parser.Results.Xmodel{:}, 'Method', parser.Results.Method);
[W, Z] = parser.Results.Statistics(X, X_k, y); % CY, output Z
S = knockoffs.select(W, q, parser.Results.Threshold);

% Reapply variable names.
if (~isempty(Xnames))
    S = Xnames(S);
end

end
