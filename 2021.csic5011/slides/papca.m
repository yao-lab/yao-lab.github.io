clear; tic;

% Parallel Analysis Program For Principal Component Analysis
%   with random normal data simulation or Data Permutations.

%  This program conducts parallel analyses on data files in which
%  the rows of the data matrix are cases/individuals and the
%  columns are variables; There can be no missing values;

%  You must also specify:
%   -- ndatsets: the # of parallel data sets for the analyses;
%   -- percent: the desired percentile of the distribution of random
%      data eigenvalues [percent];
%   -- randtype: whether (1) normally distributed random data generation 
%      or (2) permutations of the raw data set are to be used in the
%      parallel analyses (default=[2]);

%  WARNING: Permutations of the raw data set are time consuming;
%  Each parallel data set is based on column-wise random shufflings
%  of the values in the raw data matrix using Castellan's (1992, 
%  BRMIC, 24, 72-77) algorithm; The distributions of the original 
%  raw variables are exactly preserved in the shuffled versions used
%  in the parallel analyses; Permutations of the raw data set are
%  thus highly accurate and most relevant, especially in cases where
%  the raw data are not normally distributed or when they do not meet
%  the assumption of multivariate normality (see Longman & Holden,
%  1992, BRMIC, 24, 493, for a Fortran version); If you would
%  like to go this route, it is perhaps best to (1) first run a 
%  normally distributed random data generation parallel analysis to
%  familiarize yourself with the program and to get a ballpark
%  reference point for the number of factors/components;
%  (2) then run a permutations of the raw data parallel analysis
%  using a small number of datasets (e.g., 10), just to see how long
%  the program takes to run; then (3) run a permutations of the raw
%  data parallel analysis using the number of parallel data sets that
%  you would like use for your final analyses; 1000 datasets are 
%  usually sufficient, although more datasets should be used
%  if there are close calls.


% The "load" command can be used to read a raw data file
% The raw data matrix must be named "raw"
% e.g.
% 	load snp452-data.mat  %S&P500 data: 1258 daily price of 452 stocks
%   raw=diff(log(X),1);

%  These next commands generate artificial raw data 
%  (500 cases) that can be used for a trial-run of
%  the program, instead of using your own raw data; 
%  Just run this whole file; However, make sure to
%  delete these commands before attempting to run your own data.

% Start of artificial data commands.
com = randn(500,3);
raw = randn(500,9);
raw(:,1:3) = raw(:,1:3) + [ com(:,1) com(:,1) com(:,1) ];
raw(:,4:6) = raw(:,4:6) + [ com(:,2) com(:,2) com(:,2) ];
raw(:,7:9) = raw(:,7:9) + [ com(:,3) com(:,3) com(:,3) ];
% End of artificial data commands.


ndatsets  = 100  ; % Enter the desired number of parallel data sets here

percent   = 95  ; % Enter the desired percentile here

% Specify the desired kind of parellel analysis, where:
% 1 = principal components analysis
kind = 1 ;

% Enter either
%  1 for normally distributed random data generation parallel analysis, or
%  2 for permutations of the raw data set (more time consuming).
randtype = 2 ;

%the next command can be used to set the state of the random # generator
randn('state',1953125)

%%%%%%%%%%%%%%% End of user specifications %%%%%%%%%%%%%%%


[ncases,nvars] = size(raw);

evals = []; % random eigenvalues initialization
% principal components analysis & random normal data generation
if (randtype == 1)
    realeval = flipud(sort(eig(corrcoef(raw))));    % better use corrcoef
for nds = 1:ndatsets; evals(:,nds) = eig(corrcoef(randn(ncases,nvars)));end
end

% principal components analysis & raw data permutation
if (randtype == 2)
    %realeval = flipud(sort(eig(corrcoef(raw))));    % either cov/corrcoef
    realeval = flipud(sort(eig(cov(raw))));
for nds = 1:ndatsets; 
x = raw;
for lupec = 2:nvars;
    % Here we use randperm in matlabl
    x(:,lupec) = x(randperm(ncases),lupec);
    % Below is column-wise random shufflings
    %  of the values in the raw data matrix using Castellan's (1992, 
    %  BRMIC, 24, 72-77) algorithm;
    %
    %for luper = 1:(ncases -1);
    %k = fix( (ncases - luper + 1) * rand(1) + 1 )  + luper - 1;
    %d = x(luper,lupec);
    %x(luper,lupec) = x(k,lupec);
    %x(k,lupec) = d;end;end;
end
    %evals(:,nds) = eig(corrcoef(x));   % either cov/corrcoef
    evals(:,nds) = eig(cov(x));
end;end

evals = flipud(sort(evals,1));
means = (mean(evals,2));   % mean eigenvalues for each position.
evals = sort(evals,2);     % sorting the eigenvalues for each position.
percentiles = (evals(:,round((percent*ndatsets)/100)));  % percentiles.
pvals = sum(evals>(realeval*ones(1,ndatsets)),2)/ndatsets; % p-values of observed random eigenvalues greater than real eigenvalues

format short
disp([' ']);disp(['PARALLEL ANALYSIS ']); disp([' '])
if (randtype == 1);
disp(['Principal Components Analysis & Random Normal Data Generation' ]);disp([' ']);end
if (randtype == 2);
disp(['Principal Components Analysis & Raw Data Permutation' ]);disp([' ']);end
disp(['Variables  = ' num2str(nvars) ]);
disp(['Cases      = ' num2str(ncases) ]);
disp(['Datsets    = ' num2str(ndatsets) ]);
disp(['Percentile = ' num2str(percent) ]); disp([' '])
disp(['Raw Data Eigenvalues, & Mean & Percentile Random Data Eigenvalues']);disp([' '])
disp(['      Root   Raw Data   P-values    Means   Percentiles' ])
disp([(1:nvars).'  realeval  pvals means  percentiles]);


plot ( (1:nvars).',[realeval means  percentiles],'-*')
xlabel('Index'); ylabel('Eigenvalues'); title ('Real and Random-Data Eigenvalues')
set(get(gca,'YLabel'),'Rotation',90.0) % for the rotation of the YLabel
set(gca,'XTick', 1:nvars)
set(gca,'FontName','Times', 'FontSize',16, 'fontweight', 'normal' );
legend('real data eigenvalues','mean random eigenvalues',sprintf('%d%% percentile random eigenvalues',percent)); legend boxoff; 
textobj = findobj('type', 'text'); set(textobj, 'fontunits', 'points'); 
set(textobj,'FontName','Times', 'fontsize', 16); set(textobj, 'fontweight', 'normal');



disp(['time for this problem = ', num2str(toc) ]); disp([' '])


 

