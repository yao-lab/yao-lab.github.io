classdef CreateGaussian < knockoffs.tests.KnockoffTestCase
    
    methods (Test)
        function testEquiIdentity(self)
            p = 100;
            sigma = 0.4;
            Sigma = sigma * eye(p);
            diag_s = knockoffs.create.solveEqui(Sigma);
            diag_s_expected = repmat(sigma, p, 1);
            self.verifyAlmostEqual(diag_s, diag_s_expected);
        end
        function testEquiToeplitz(self)
            p = 100;
            rho = 0.5;
            Sigma = toeplitz(rho.^(0:(p-1)));
            diag_s = knockoffs.create.solveEqui(Sigma);
            diag_s_expected = repmat(min(1,2*eigs(Sigma,1,'SM')), p, 1);
            self.verifyAlmostEqual(diag_s, diag_s_expected);
        end

        function testSDP(self)
            diag_Sigma = 0.1:0.1:4;
            Sigma = diag(diag_Sigma(randperm(length(diag_Sigma))));
            diag_s = sparse(diag(knockoffs.create.solveSDP(Sigma)));
            diag_s_expected = sparse(Sigma);
            self.verifyAlmostEqual(diag_s, diag_s_expected);
        end

        function testEquiCov(self)
            p = 5;
            rho = 0.5;
            Sigma = toeplitz(rho.^(0:(p-1)));
            diag_s = sparse(diag(knockoffs.create.solveEqui(Sigma)));
            diag_s_expected = sparse(diag(repmat(min(1,2*eigs(Sigma,1,'SM')), p, 1)));
            n = 10000000;
            mu = randn(1,p);
            X = mvnrnd(mu, Sigma, n);
            X_k = knockoffs.create.gaussian_sample(X, mu, Sigma, diag_s);
            G = [Sigma, Sigma-diag_s_expected; Sigma-diag_s_expected, Sigma];
            Delta = abs(G - cov([X,X_k]));
            self.verifyLessThan(max(Delta(:)),1e-2);
        end
        
        function testSDPCovSmall(self)
            p = 5;
            rho = 0.2;
            Sigma = toeplitz(rho.^(0:(p-1)));
            diag_s = sparse(diag(knockoffs.create.solveSDP(Sigma)));
            diag_s_expected = sparse(eye(p));
            n = 10000000;
            mu = randn(1,p);
            X = mvnrnd(mu, Sigma, n);
            X_k = knockoffs.create.gaussian_sample(X, mu, Sigma, diag_s);
            G = [Sigma, Sigma-diag_s_expected; Sigma-diag_s_expected, Sigma];
            Delta = abs(G - cov([X,X_k]));
            self.verifyLessThan(max(Delta(:)),1e-2);
        end

        function testSDPCovLarge(self)
            p = 10;
            Sigma = corr(randn(2*p,p));
            diag_s = sparse(diag(knockoffs.create.solveSDP(Sigma)));
            n = 1000000;
            mu = randn(1,p);
            X = mvnrnd(mu, Sigma, n);
            X_k = knockoffs.create.gaussian_sample(X, mu, Sigma, diag_s);
            iden = diag(ones(2*p-abs(p),1),p) + diag(ones(2*p-abs(p),1),-p);
            G = [Sigma, Sigma; Sigma, Sigma];
            G(iden~=0) = 0;            
            Ghat = cov([X,X_k]);            
            Ghat(iden~=0) = 0;
            Delta = abs(G - Ghat);
            self.verifyLessThan(max(Delta(:)),1e-2);
        end
       end
    
end