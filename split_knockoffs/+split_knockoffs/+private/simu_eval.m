function[fdr, power] = simu_eval(gamma_true, result)
% this function calculate the FDR and Power when gamma^* is available, i.e.
% in simulations.

% input argument
% gamma_true: true signal of gamma
% result: the estimated support set of gamma

% output argument
% fdr: false discovery rate of the estimated support set
% power: power of the estimated support set

if isempty(result) == true
    fdr = 0;
else
    total_posi = length(result);
    false_posi = total_posi - sum(gamma_true(result) ~= 0);
    fdr = false_posi / total_posi;
end

power = sum(gamma_true(result) ~= 0) / sum(gamma_true ~= 0);

end