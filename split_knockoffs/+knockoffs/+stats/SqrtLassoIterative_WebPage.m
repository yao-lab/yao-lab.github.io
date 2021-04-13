function [betaSQ, sSQ] = SqrtLassoIterative_WebPage(X, y, lambda, gamma,varargin)
% This function computes the square-root lasso estimator.
% Available from: https://faculty.fuqua.duke.edu/~abn5/belloni-software.html
%
% input:
% X design matrix     n x p
% y response variable n x 1
% lambda              penalty parameter 
% gamma               p x 1  (vector of loadings typically: gamma_j = sqrt( En[x_{ij}^2] ) 
%
% min_{beta}  sqrt( Qhat(beta) ) + (lambda/n)*\sum_j gamma_j*|beta_j|
%
% output:   betaSQ  square root LASSO estimator
%           sQR     number of non zero components in betaSQ
[MaxIter,PrintOut,OptTolNorm,OptTolObj] = process_options(varargin,'MaxIter',10000,'PrintOut',1,'OptTolNorm',1e-6,'OptTolObj',1e-8);

[n,p] = size(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Initial Point: start from the Ridge estimator
RidgeMatrix = eye(p);
for j = 1 : p 
    RidgeMatrix(j,j) = (lambda)*gamma(j);
end
beta = (X'*X + RidgeMatrix) \ (X'*y);


% Printout 
if ( PrintOut >= 1)    
    fprintf('%12s %16s %12s %17s %10s %10s\n','Iter','sqrt(Qhat(beta))','||beta||_1','||beta-beta_old||','primal','dual');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Inirialization
Iter = 0;                      %iteration count

XX = X'*X/n;                % Gram matrix
Xy = X'*y/n;                % 

ERROR = y - X*beta;         % residuals
Qhat = sum( ERROR.^2 )/n;   % average of squared residuals


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Main Loop
while Iter < MaxIter
           
    Iter = Iter + 1;   
    beta_old = beta;
    
    %%%% Go over each coordinate
    for j = 1:p
               
        % Compute the Shoot and Update the variable
        S0 = XX(j,:)*beta - XX(j,j)*beta(j) - Xy(j);

        % ERROR = y - X*beta +  X(:,j)*beta(j);
        if ( abs(beta(j)) > 0 )
            ERROR = ERROR + X(:,j)*beta(j);
            Qhat = sum( ERROR.^2 )/n;
        end
        
        %%% Note that by C-S
        %%% S0^2 <= Qhat * XX(j,j), so that  Qhat >= S0^2/XX(j,j)  :)

        if ( n^2 < (lambda * gamma(j))^2 / XX(j,j) )
        
            beta(j) = 0; 
        
        elseif S0 > (lambda/n) * gamma(j) * sqrt(Qhat)
            %%% Optimal beta(j) < 0
            % For lasso: beta(j) = (lambda - S0)/XX2(j,j); 
            % for suqare-root lasso
            beta(j) = (  ( lambda * gamma(j) / sqrt( n^2 - (lambda * gamma(j))^2 / XX(j,j)  ) )  *  sqrt( max(Qhat - (S0^2/XX(j,j)),0) )  - S0 )   /   XX(j,j);
        
            ERROR = ERROR - X(:,j)*beta(j);
            
        elseif S0 < - (lambda/n) * gamma(j) * sqrt(Qhat)
            %%% Optimal beta(j) > 0
            % For lasso: beta(j) = (-lambda - S0)/XX2(j,j);
            % for square-root lasso
            beta(j) = (  -  ( lambda * gamma(j) / sqrt( n^2 - (lambda * gamma(j))^2 / XX(j,j)  ) )  *  sqrt( max( Qhat - (S0^2/XX(j,j)),0) )  - S0 )   /   XX(j,j);
            ERROR = ERROR - X(:,j)*beta(j);
            
        elseif abs(S0) <= (lambda/n) * gamma(j) * sqrt(Qhat)
            %%% Optimal beta(j) = 0
            beta(j) = 0;
        end
        
    end
    
    
    %%% Update primal and dual value
    fobj = sqrt( sum((X*beta-y).^2)/n )  +  (lambda*gamma/n)'*abs(beta);
        
    if ( norm(ERROR) > 1.0e-10 )
            aaa  = (sqrt(n)*ERROR/norm(ERROR));
            dual = aaa'*y/n - abs(  lambda*gamma/n - abs(X'*aaa/n) )'*abs(beta);
    else
            dual = lambda*gamma'*abs(beta)/n;
    end
       
    
    % Print Current Information        
    if ( PrintOut >= 1)   
        fprintf('%12d %12.2e %12.2e %12.2e %12.2e %12.2e\n',Iter,sqrt( sum((X*beta-y).^2)/n ), sum(abs(beta)),sum(abs(beta-beta_old)),fobj, dual );
    end   

    % Stopping Criterion
    if   ( sum(abs(beta-beta_old)) < OptTolNorm )
        
            if ( fobj - dual  < OptTolObj  )
            
                break;
    
            end

    end
    
    
end

betaSQ = beta;
sSQ = sum( abs(betaSQ) > 0 );

if ( PrintOut > 0 )
    fprintf('Number of iterations: %d \n',Iter);
    fprintf('Number of Nonzero components: %d\n',sSQ);

    if ( Qhat > 1.0e-10 )
        fprintf('Maximal Dual violation: %15.2e (max relative: %5.2e)\n', max( 0, max(  abs(X'*aaa/n) - lambda*gamma/n  ) ), max( 0, max( (abs(X'*aaa/n) - lambda*gamma/n  )./(lambda*gamma/n) ) ) );
    end

end

