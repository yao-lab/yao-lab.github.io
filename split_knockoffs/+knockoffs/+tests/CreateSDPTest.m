classdef CreateSDPTest < knockoffs.tests.KnockoffTestCase
    
    methods (Test)
        function testCovariances(self)
            X = knockoffs.private.normc(randn(20,10));
            X_ko = knockoffs.create.fixed_SDP(X, false);
            self.verifyCovariances(X, X_ko);
        end

        function testRandomizedCovariances(self)
            X = knockoffs.private.normc(randn(20,10));
            X_ko = knockoffs.create.fixed_SDP(X, true);
            self.verifyCovariances(X, X_ko);
        end

        function testDimensionCheck(self)
            X = knockoffs.private.normc(randn(10,10));
            self.verifyError(@() knockoffs.create.fixed_SDP(X), ...
                'knockoff:DimensionError')
        end
        
        function testPermutationInvariance(self)
            X = knockoffs.private.normc(randn(20,10));
            I = randperm(10);
            X_ko = knockoffs.create.fixed_SDP(X, false);
            X_perm_ko = knockoffs.create.fixed_SDP(X(:,I), false);
            self.verifyEqual(X_ko(:,I), X_perm_ko, 'AbsTol', 1e-4);
        end
        
        function testDiagonalSigmaCorrectness(self)
            diag_Sigma = 0.1:0.1:10;
            Sigma = diag(diag_Sigma);
            s_sdp = knockoffs.create.solveSDP(Sigma);
            s_sdp = reshape(s_sdp, size(diag_Sigma));
            self.verifyAlmostEqual(diag_Sigma, s_sdp);
        end
    end
    
    methods
        function verifyCovariances(self, X, X_ko)
            G = X'*X;
            self.verifyAlmostEqual(X_ko'*X_ko, G);
            self.verifyAlmostEqual(offdiag(X'*X_ko), offdiag(G))
            self.verifyLessThan(diag(X'*X_ko), 1 + 1e-5);
        end
    end

end

function B = offdiag(A)
B = A - diag(diag(A));
end