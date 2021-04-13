%if ~exist('glmnet-matlab','dir')
%    return('Did not find glmnet-matlab!')

if ~exist('split_knockoffs','dir')
    cd ..
    addpath(pwd)
    cd split_knockoffs
end