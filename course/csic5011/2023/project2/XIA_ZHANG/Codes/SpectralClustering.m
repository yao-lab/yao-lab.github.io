function [c,Acc] = SpectralClustering(A,c0)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

d = sum(A,2);
D = diag(d);
L = D-A;
Normalized_L = D^(-1/2)*L*D^(-1/2);

%% Cheeger vector + equally cut
[V,~] = eigs(Normalized_L);
[~,n] = size(V);
v2 = V(:,n-1);

% c(1) = 0
[n,~] = size(A);
c = zeros(n,1);
if v2(1) > median(v2)
    c(v2 > median(v2))=0;
    c(v2 <= median(v2))=1;
else
    c(v2 > median(v2))=1;
    c(v2 <= median(v2))=0;
end    

% accuracy  
Acc = length(find(c==c0))/n;

end

