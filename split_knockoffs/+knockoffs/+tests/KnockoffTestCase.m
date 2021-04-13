classdef KnockoffTestCase < matlab.unittest.TestCase
    
    methods
        function verifyAlmostEqual(self, actual, expected, rtol)
            if ~exist('rtol', 'var')
                rtol = 1e-5;
            end
            self.verifyEqual(actual, expected, 'RelTol', rtol)
        end
    end
    
end