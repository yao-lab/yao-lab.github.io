function [L, R, IDX, C, DL]=kcenter(X,k,L0,EorD)
% Farthest-First Traversal Algorithm as a 2-approximation for 
% kcenter clustering
%   [L,R,IDX,C,DL] = KCENTER(X,k,L0,EorD)
%
% INPUT:
%   X - see description for input EorD.
%   k - the number of centers to be chosen.
%   L0 - the first centroid index.
%   EorD - character EorD determines how input matrix is interpreted. If 
%       EorD is 'e', then the N x p input matrix X is interpreted as N 
%       points in R^p. If EorD is 'd', then the N x N input matrix X is 
%       interpreted as the distance matrix for N points in an arbitrary 
%       metric space.
%
% OUTPUT:
%   L - an p-by-1 vector containing indices of each landmark.
%   R - covering radius, i.e. the smallest number such that every data 
%       point lies within distance R of a landmark point.
%   IDX - an N-by-1 vector containing the cluster indices of each point.
%   C - a k-by-p matrix for the k cluster centroid locations.
%   DL - an N-by-k matrix of distances from each point to every centroid. 
%
% NOTES:
%   This m-file uses m-file px_maxmin from the C++ version of JPlex.
%
% REFERENCE: 
%   T.F. Gonzalez. Clustering to minimize the maximum intercluster
%   distance. Theoretical Computer Science, 38:293-306, 1985.
%

%   Yuan Yao PKU, 2011.02.25

if (nargin <= 2),
    L0 = 1;
    EorD = 'e';
elseif (nargin == 3),
    EorD = 'e';
end

if EorD=='e'
    [L,ld,r]=px_maxmin(X','vector',k,'n',L0,'seeds');    
elseif EorD=='d'
    [L,ld,r]=px_maxmin(X','metric',k,'n',L0,'seeds');
else
    error('input EorD must be either character e or character d')
end

L = reshape(L,[k,1]);

if (nargout >= 2)
  R = r;
end

if (nargout >= 3)
    [rnn,I]=min(ld',[],2);
    IDX = I;
end

if (nargout >= 4)
     C = X(L,:);
end
  
if (nargout >= 5)
     DL = ld';
end


function [L, DL, R] = px_maxmin(varargin)

%PX_MAXMIN -- select landmark points by greedy optimisation
%
% Given a set of N points, PX_MAXMIN selects a subset of n points called
% 'landmark' points by an interative greedy optimisation. Specifically,
% when j landmark points have been chosen, the (j+1)-st landmark point
% maximises the function 'minimum distance to an existing landmark point'.
%
% The initial landmark point is arbitrary, and may be chosen randomly or
% by decree. More generally, the process can be 'seeded' by up to k
% landmark points chosen randomly or by decree.
%
% The input data can belong to one of the following types:
%
%    'vector': set of d-dimensional Euclidean points passed to the
%              function as d-by-N matrix
%
%    'metric': N-by-N matrix of distances
%
%    'rows': function handle with one vector argument I=[i1,i2,...,ip]
%              which returns a p-by-N matrix of distances between the p
%              specified points and the full data set.
%
% The output L is a list of indices for the landmark points, presented in
% the order of discovery. User-specified seeds are listed first, followed
% by randomly chosen seeds.
%
% A second output argument, DL, returns the n-by-N matrix of distances
% between landmark points and all data points.
%
% A third output argument, R, returns the covering number of the landmark
% set. R is the smallest number such that every data point lies within
% distance R of a landmark point.
%
% Syntax:
%
% L          = PX_MAXMIN(data1, type1, data2, type2, ...);
% [L, DL]    = PX_MAXMIN(data1, type1, data2, type2, ...);
% [L, DL, R] = PX_MAXMIN(data1, type1, data2, type2, ...);
%
% Each pair (data, type) is one of the following:
%
%  (X, 'vector'), (D, 'metric'), (fD,'rows')
%
%  (n, 'n')     -- number of landmarks
%
%  (S, 'seeds') -- list of seeds specified by the user; no repeats are
%                  allowed.
%
%  (k, 'rand')  -- number of randomly chosen seeds
%
% The pairs may be listed in any order. The type strings are sensitive to
% case. Exactly one of {X, D, fD} must be specified, and n must always be
% specified.
%
% Both of {k, S} are optional arguments. If S is specified, and is not
% the empty list, then k=0 is the default. Otherwise, the defaults are
% k=1, S=[].
%
% The function PX_LANDMARKD is a C++ version of PX_MAXMIN and may be used
% instead. The syntax and functionality are slightly different.
%
%Plex Metric Data Toolbox version 2.5 by Vin de Silva, Patrick Perry and
%contributors. See PX_PLEXINFO for credits and licensing information.
%Released with Plex version 2.5. [2006-Jul-14]

% [2004-May-06] Modifications:
%               -replacing @local_euclid with @local_euclid2
%               -returning R as an output argument
%
% [2004-May-24] -speed up 'min' step using incremental updates.
%               -R now returns the *next* value of MaxMin.
%
% [2004-Apr-28] Vin de Silva, Department of Mathematics, Stanford.

%----------------------------------------------------------------
% collate the input variables
%----------------------------------------------------------------

types = {'vector', 'metric', 'rows', 'n', 'seeds', 'rand'};

if any(~ismember(varargin(2:2:end), types))
  error('Invalid data type specified.')
end

if (nargin ~= (2 * length(unique(varargin(2:2:end)))))
  error('Repeated or missing data types.')
end

[isused, input_ix] = ismember(types, varargin(2: 2: end));

if (sum(isused(1:3)) ~= 1)
  error('Points must be specified in exactly one of the given formats.')
end

if ~isused(4)
  error('Number of landmarks must be specifed.')
end

%----------------------------------------------------------------
% standardize the input
%----------------------------------------------------------------

%------------------------------------------------
% By the end of this section, feval(Dfun,list) 
% will return D(list,:), whichever the input
% format of the data
%------------------------------------------------

dataformat = find(isused(1:3)); % which format have the data been
                                % entered in?
switch dataformat
 case 1 % 'vector'
  X = varargin{2*input_ix(1)-1};
  [d,N] = size(X);

  %Dfun = @local_euclid;
  
  L2sq = sum(X.^2,1);    % these values are repeatedly used:
  Dfun = @local_euclid2; % optimised to take advantage of this

 case 2 % 'metric'
  D = varargin{2*input_ix(2)-1};
  N = length(D);
  Dfun =@local_submatrix;
  
 case 3 % 'rows'
  Dfun = varargin{2*input_ix(3)-1};
  N = size(feval(Dfun,1), 2);
  
end

%------------------------------------------------
% collect n
%------------------------------------------------
n = varargin{2*input_ix(4)-1};


%------------------------------------------------
% determine seeding
%------------------------------------------------
if isused(5)
  S = varargin{2*input_ix(5)-1};
  if (length(S) ~= length(unique(S)))
    error('S must not contain repeated elements.')
  end
else
  S = [];
end

if isused(6)
  k = varargin{2*input_ix(6)-1};
elseif isempty(S)
  k = 1;
else
  k = 0;
end


%----------------------------------------------------------------
% main loop
%----------------------------------------------------------------
s = length(S);
if (n < s + k)
  error('Too many seeds!')
end

% read in seed points from S
L = zeros(1,n);
L(1: s) = S;

unused = setdiff((1:N), S);
if (length(unused) ~= (N-s))
  error('Seeds specified incorrectly.')
end
  
% generate random seed points
foo = randperm(N-s);
randseeds = unused(foo(1:k));
clear foo

L(s+1: s+k) = randseeds;

% generate remaining landmarks by maxmin
DD = zeros(n, N);
DD((1: s+k), :) = feval(Dfun, L(1: s+k));

DDmin = min(DD((1:s+k),:), [], 1);
r = zeros(n-s-k,1);
for a = (s+k+1: n)
  
  [r(a-1), newL] = max(DDmin, [], 2);
  L(a) = newL;
  DD(a,:) = feval(Dfun, newL);
  DDmin = min(DDmin, DD(a,:));
  
end
r(n) = max(DDmin);

%----------------------------------------------------------------
% finish!
%----------------------------------------------------------------

if (nargout >= 2)
  DL = DD;
end
  
if (nargout >= 3)
  R = r;
end

return
%----------------------------------------------------------------
% local functions
%----------------------------------------------------------------
function DD = local_euclid(list);
X = evalin('caller', 'X');
[d,N] = size(X);

DD = sqrt(max(0, repmat(sum(X(:,list).^2, 1)', [1 N]) ...
	       + repmat(sum(X.^2, 1), [length(list) 1]) ...
	       - 2 * X(:, list)' * X));

%----------------------------------------------------------------
function DD = local_euclid2(list);
X = evalin('caller', 'X');     
[d,N] = size(X);
L2sq = evalin('caller','L2sq'); 

DD = sqrt(max(0, repmat(L2sq(list)', [1 N]) ...
	       + repmat(L2sq, [length(list) 1]) ...
	       - 2 * X(:, list)' * X));

%----------------------------------------------------------------
function DD = local_submatrix(list);
D = evalin('caller', 'D');
DD = D(list, :);

%----------------------------------------------------------------
