function [W,Z] = bvsProbDiff2(X, X_ko, y, options)
% KNOCKOFFS.STATS.BVSPROBDIFF2  The Bayesian posterior probability 
%  difference statistic W (with additional constraint)
%   [W, Z] = KNOCKOFFS.STATS.BVSPROBDIFF2(X, X_ko, y)
%   [W, Z] = KNOCKOFFS.STATS.BVSPROBDIFF2(X, X_ko, y, options)
%
%   Computes the statistic
%
%     W_j = |P_j| - |\tilde P_j|,
%
%   where P_j and \tilde P_j are the posterior probabilities of the jth
%   variable and its knockoff being nonzero in the Bayesian variable
%   selection problem with conjugate priors (for multivariate normal X).
%   Requires Gamma_j + \tilde Gamma_j <= 1
%
% See also KNOCKOFFS.STATS.BVSPROBDIFF2

if ~exist('options', 'var')
  options = struct();
  options.tau2 = 1;
  options.pi = 0.02; %this is a fraction of p, not 2p
  options.alpha = 2; %shape
  options.delta = 1; %scale=beta on Wikipedia
  options.burn = 1000;
  options.nit = 3000;
end

tau2 = options.tau2;
pi = options.pi;
alpha = options.alpha;
delta = options.delta;
burn = options.burn;
nit = options.nit;

[n,p] = size(X);
Xall = [X X_ko];

sig2 = nan(burn+nit,1);
g = nan(2*p,burn+nit);
b = nan(2*p,burn+nit);

sig2(1) = delta/(alpha-1);
g(:,1) = zeros(2*p,1);
b(:,1) = zeros(2*p,1);

X2 = sum(Xall.^2,1);
XtX = Xall'*Xall;
r = y; %residual vector
Xr = Xall'*r;
%tic
for i = 2:(burn+nit)
  % beta updates
  for j = 1:(2*p)
    if g(j,i-1)==0
      b(j,i) = normrnd(0,sqrt(tau2));
    else
      b(j,i) = normrnd((Xr(j)+X2(j)*b(j,i-1))/(X2(j)+sig2(i-1)/tau2), sqrt(sig2(i-1)/(X2(j)+sig2(i-1)/tau2)));
      Xr = Xr - XtX(:,j)*(b(j,i)-b(j,i-1));
    end
  end
  r = r - Xall(:,g(:,i-1)==1)*(b(g(:,i-1)==1,i)-b(g(:,i-1)==1,i-1));
  
  % gamma updates
  for j = 1:p
    if g(j,i-1)==0 && g(j+p,i-1)==0
      r0 = r; r1 = r-Xall(:,j)*b(j,i); r2 = r-Xall(:,j+p)*b(j+p,i);
    elseif g(j,i-1)==1 && g(j+p,i-1)==0
      r0 = r+Xall(:,j)*b(j,i); r1 = r; r2 = r0-Xall(:,j+p)*b(j+p,i);
    else %g(j,i-1)==0 && g(j+p,i-1)==1
      r0 = r+Xall(:,j+p)*b(j+p,i); r1 = r0-Xall(:,j)*b(j,i); r2 = r;
    end
    logpr0 = log(1-pi) - sum(r0.^2)/(2*sig2(i-1));
    logpr1 = log(pi/2) - sum(r1.^2)/(2*sig2(i-1));
    logpr2 = log(pi/2) - sum(r2.^2)/(2*sig2(i-1));
    choicej = find(mnrnd(1,[1/(1+exp(logpr1-logpr0)+exp(logpr2-logpr0)) 
                            1/(exp(logpr0-logpr1)+1+exp(logpr2-logpr1)) 
                            1/(exp(logpr0-logpr2)+exp(logpr1-logpr2)+1)]));
    switch choicej
      case 1
        g(j,i) = 0; g(j+p,i) = 0; r = r0;
      case 2
        g(j,i) = 1; g(j+p,i) = 0; r = r1;
      case 3
        g(j,i) = 0; g(j+p,i) = 1; r = r2;
    end
  end
  dg = g(:,i)~=g(:,i-1);
  Xr = Xr - XtX(:,dg)*(b(dg,i).*(g(dg,i)-g(dg,i-1)));
  
  % sigma2 updates
  sig2(i) = 1/gamrnd(alpha+n/2,1/(delta+sum(r.^2)/2));

%   subtime = toc;
%   if i==burn, fprintf(['Burn-in finished after ' num2str(subtime) 'sec\n']); end
%   if i>burn && mod(i-burn,nit/2)==0, fprintf(['Iteration ' num2str(i-burn) ' complete after ' num2str(subtime) '\n']); end
end
Z = mean(g(:,burn+(1:nit)),2);

W = abs(Z(1:p))-abs(Z((p+1):(2*p)));

end