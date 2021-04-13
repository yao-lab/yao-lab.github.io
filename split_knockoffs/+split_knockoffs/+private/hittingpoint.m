function[Z, r] = hittingpoint(coef, lambdas)
% private.hittingpoint calculate the hitting time and the sign of
% respective variable in a path. 

% input argument
% coef: the path for one variable
% lambdas: respective value of lambda in the path

% output argument
% Z: the hitting time
% r: the sign of respective variable at the hitting time


n_lambda = length(lambdas);

Z = 0;
r = 0;

% calculate Z and r
for j = 1: n_lambda
    if abs(coef(j)) ~= 0
        Z = lambdas(j);
        r = sign(coef(j));
        break
    end
end
end