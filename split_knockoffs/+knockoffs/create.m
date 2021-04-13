function X_k = create(X, model, varargin)
% KNOCKOFFS.CREATE  Create knockoffs given the model parameters and a 
% matrix of observations for the original variables.
%
%   X_k = KNOCKOFFS.CREATE(X, model, {model parameters}, ...)
%
%   By default, creates Gaussian sdp-correleated knockoffs.
%
%   Inputs:
%      X    - n x p (scaled) data matrix
%             Requires (n >= 2*p) if model='fixed'
%    model  - 'gaussian' for Model-X Gaussian variables or
%             'fixed' for Fixed-X variables
%             Default: 'gaussian'
%
%   Required model-specific inputs:
%      Mu   - p x 1 mean vector. Required if model='gaussian'
%     Sigma - p x p covariance matrix. Required if model='gaussian' 
%   
%   Optional Inputs:
%       'Method' - Method to use for creating knockoffs.
%
%   Optional model-specific inputs:
%     'Randomize' - whether to use randomization in the construction of
%                   the knockoff variables (if model='fixed')
%
%   Outputs:
%       X_k - n x p knockoff variable matrix
%
%   See also KNOCKOFFS.CREATE.GAUSSIAN, KNOCKOFFS.CREATE.FIXED, KNOCKOFFS.SELECT.

if ~exist('model', 'var') || isempty(model), model = 'gaussian'; end;

parser = inputParser;
parser.CaseSensitive = false;
if (~verLessThan('matlab', '8.2')) % R2013b or later
    parser.PartialMatching = false;
end
istable_safe = @(x) ~verLessThan('matlab', '8.2') && istable(x);

parser.addRequired('X', @(x) isnumeric(x) || istable_safe(x));
parser.addRequired('model', @isstr);

model = lower(model);
switch model
    case 'gaussian'
        parser.addRequired('Mu', @isnumeric);
        parser.addRequired('Sigma', @(x) isnumeric(x) && all(eig(x) > 0));
        parser.addOptional('Method', 'sdp');
        parser.parse(X, model, varargin{:});
        sampleK = @knockoffs.create.gaussian;
        X_k = sampleK(X,parser.Results.Mu, parser.Results.Sigma, parser.Results.Method);
    case 'fixed'
        parser.addOptional('Method', 'sdp');
        parser.addOptional('Randomize', []);
        parser.parse(X, model, varargin{:});
        sampleK = @knockoffs.create.fixed;
        X_k = sampleK(X, parser.Results.Method, parser.Results.Randomize);
    otherwise
        error('Invalid variables model %s', model)
end

end
