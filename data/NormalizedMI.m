function [NMI] = NormalizedMI(trueLabel, partitionMatrix)
% normalized mutual information
% Author: Weike Pan, weikep@cse.ust.hk
% Ref: Dhilon, KDD 2004 Kernel k-means, Spectral Clustering and Normalized Cuts
% Section 6.3
% High NMI value indicates that the clustering and true labels match well 

% usage: NormalizedMI([1 1 1 2 2 2]', [1 2 1 3 3 3]')

%%
truey = trueLabel;
[m1, c] = size(truey); % c: class #

PM = partitionMatrix;
[m2, k] = size(PM); % k: cluster #

%%
% check whether m1 == m2
if m1 ~= m2
    error('m1 not equal m2');    
else
    m = m1;
end

%% change the truelable or the partition matrix: m \times c
if c == 1
    c = length( unique(truey) );
    tmp = zeros(m,c);
    for i = 1 : c
        tmp((truey == i), i) = 1;
    end
    truey = tmp;    
end

if k == 1
    k = length( unique(PM) );
    tmp = zeros(m,k);
    for i = 1 : k
        tmp((PM == i), i) = 1;
    end
    PM = tmp;    
end

%%

% *****************************
% calculate the confusion matrix
for l = 1 : 1 : k  
    for h = 1 : 1 : c
        n(l,h) = sum( (truey(:,h) == 1) & (PM(:,l) == 1) );    
    end
end



% *****************************
NMI = 0;
for l = 1 : 1 : k
    
    for h = 1 : 1 : c
        NMI = NMI + (n(l,h)/m) * log(  ( n(l,h)*m + eps) / ( sum(n(:,h))*sum(n(l,:)) + eps) ); 
    end

end

Hpi = - sum( (sum(PM)/m) .* log( sum(PM)/m + eps ) );
Hvarsigma = - sum( (sum(truey)/m) .* log( sum(truey)/m + eps ) );

% NMI = 2*NMI/(Hpi + Hvarsigma);

% JMLR03, A. Strehl and J. Ghosh. Cluster ensembles -- a knowledge reuse framework for combining multiple partitions.
NMI = NMI/sqrt(Hpi*Hvarsigma);







