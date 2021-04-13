classdef CreateEquiTest < knockoffs.tests.KnockoffTestCase
    
    methods (Test)
        function testCovariances(self)
            X = knockoffs.private.normc(randn(20,10));
            X_ko = knockoffs.create.fixed_equi(X, false);
            self.verifyCovariances(X, X_ko);
        end

        function testRandomizedCovariances(self)
            X = knockoffs.private.normc(randn(20,10));
            X_ko = knockoffs.create.fixed_equi(X, true);
            self.verifyCovariances(X, X_ko);
        end

        function testDimensionCheck(self)
            X = knockoffs.private.normc(randn(10,10));
            self.verifyError(...
                @() knockoffs.create.fixed_equi(X), ...
                'knockoff:DimensionError')
        end
        
        function testPermutationInvariance(self)
            X = knockoffs.private.normc(randn(20,10));
            I = randperm(10);
            X_ko = knockoffs.create.fixed_equi(X, false);
            X_perm_ko = knockoffs.create.fixed_equi(X(:,I), false);
            self.verifyAlmostEqual(X_ko(:,I), X_perm_ko)
        end
    end
    
    methods
        function verifyCovariances(self, X, X_ko)
            G = X'*X;
            s = min(2*min(eig(G)), 1);
            s = repmat(s, [1, size(X,2)]);
            self.verifyAlmostEqual(X_ko'*X_ko, G);
            self.verifyAlmostEqual(X'*X_ko, G - diag(s));
        end
    end

end