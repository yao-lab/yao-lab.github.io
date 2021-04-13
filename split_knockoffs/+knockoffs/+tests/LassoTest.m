classdef LassoTest < knockoffs.tests.KnockoffTestCase
    
    methods (Test)
        % Test the special case of an orthonormal design (X'X = I),
        % in which case the lasso minimization problem
        %
        %   (1/2) ||y - X*beta||_2^2 + lambda * ||beta||_1
        %
        % has the closed form solution
        %
        %     beta = sgn(beta_LS) max(0, abs(beta_LS) - lambda)),
        %
        % where beta_LS is the ordinary least-squares solution.
        function testOrthonormal(self)
            n = 10; p = 5; sigma = 1e-2;
            X = randn(n, p);
            X = orth(bsxfun(@minus,X,mean(X,1)));
            beta = randn(p,1);
            y = X*beta + sigma .* randn(n,1);
            
            betaLS = X'*y;
            nlambda = 10000;
            lambdaMax = knockoffs.stats.private.lassoMaxLambda(X, y, nlambda);
            self.verifyAlmostEqual(lambdaMax, abs(betaLS'), 1e-3);
        end
        
        % Test that lassoMaxLambda is invariant under permutation.
        function testPermutationInvariance(self)
            n = 100; p = 30; k = 5; sigma = 1e-2;
            X = knockoffs.private.normc(randn(n, p));
            beta = zeros(p,1);
            beta(randsample(p, k)) = 1;
            y = X*beta + sigma .* randn(n,1);
            
            I = randperm(p);
            nlambda = 10000;
            lambdaMax = knockoffs.stats.private.lassoMaxLambda(X, y, nlambda);
            lambdaMaxPerm = knockoffs.stats.private.lassoMaxLambda(X(:,I), y, nlambda);
            self.verifyAlmostEqual(lambdaMax(I), lambdaMaxPerm, 1e-2);
            
            [~,path] = sort(lambdaMax(I), 'descend');
            [~,pathPerm] = sort(lambdaMaxPerm, 'descend');
            self.verifyEqual(path, pathPerm);
        end
    end

end