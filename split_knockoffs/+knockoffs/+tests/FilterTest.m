classdef FilterTest < knockoffs.tests.KnockoffTestCase
    % Test the main entry point for this package.
    
    methods (Test)
        % Test whether the filter is invariant under permutations of
        % the columns of the design matrix.
        function testPermutationInvarianceFixed(self)
            n = 100; p = 50; k = 5; q = 0.20;
            X = randn(n, p);
            beta = zeros(p, 1);
            beta(randsample(p,k)) = 3.5;
            y = X*beta + randn(n,1);
            
            X = array2table(X);
            I = randperm(p);
            S = knockoffs.filter(X, y, q, {'fixed'});
            S_perm = knockoffs.filter(X(:,I), y, q, {'fixed'});
            self.verifyEqual(sort(S), sort(S_perm));
        end

        function testPermutationInvarianceGaussian(self)
            n = 100; p = 50; k = 5; q = 0.20; rho = 0.5;
            Sigma = toeplitz(rho.^(0:(p-1)));
            mu = randn(1,p);
            X = mvnrnd(mu, Sigma, n);
            beta = zeros(p, 1);
            beta(randsample(p,k)) = 3.5;
            y = X*beta + randn(n,1);
            
            stats = @knockoffs.stats.lassoLambdaSignedMax;
            X = array2table(X);
            I = randperm(p);
            seed = randi(100000);
            rng(seed);
            S = knockoffs.filter(X,y, q, {'gaussian',mu,Sigma},'Method','equi', 'Statistics',stats);
            rng(seed);
            S_perm = knockoffs.filter(X(:,I), y, q, {'gaussian',mu,Sigma},'Method','equi', 'Statistics',stats);
            self.verifyEqual(sort(S), sort(S_perm));
        end
             
    end

end
