% load data
clear all
clc
load('karate.mat')
n = length(c0);

% Spectral clustering
[c_spectral,Acc_spectral] = SpectralClustering(A,c0);
coach_equal = find(c_spectral==c_spectral(1));   

% Transition Path
[c_TPT,J_plus,T,Acc_TPT] = TPT(A,c0);
coach_TPT = find(c_TPT==c_TPT(1));

% Perturbation Analysis
G = graph(A);
edges = G.Edges.EndNodes;
[num_edges,~] = size(edges);

% Missing Edges
K = 7;
MaxIters = 500;
for k = 1:K
    Iters = 0;
    for i = 1:MaxIters
        p = randperm(num_edges,k); % determine the idx of deleted edges
        H = rmedge(G,p);
        A_missing = full(adjacency(H));
        d = sum(A_missing,2);
        if  ismember(0,d)
            continue;    % Let H still be connected
        else
            Iters = Iters+1;
        end
        [~,Acc_spectral] = SpectralClustering(A_missing,c0); 
        [~,~,~,Acc_TPT] = TPT(A_missing,c0);        
        Acc_missing_spectral(Iters) = Acc_spectral;
        Acc_missing_TPT(Iters) = Acc_TPT;
    end 
    mean_spectral(k) = mean(Acc_missing_spectral);
    std_spectral(k) = std(Acc_missing_spectral);
    mean_TPT(k) = mean(Acc_missing_TPT);
    std_TPT(k) = std(Acc_missing_TPT);
end


% Noisy Edges
K = 7;
MaxIters = 500;
for k = 1:K
    Iters = 0;
    for i = 1:MaxIters
        % determine the idx of noisy edges
        s = randperm(n,k); 
        t = randperm(n,k);
        H_noisy = addedge(G,s,t,ones(size(s)));
        A_noisy = full(adjacency(H_noisy));
        d = sum(A_noisy,2);
        if  ismember(0,d)
            continue;    % Let H still be connected
        else
            Iters = Iters+1;
        end
        [~,Acc_spectral] = SpectralClustering(A_noisy,c0); 
        [~,~,~,Acc_TPT] = TPT(A_noisy,c0);        
        Acc_noisy_spectral(Iters) = Acc_spectral;
        Acc_noisy_TPT(Iters) = Acc_TPT;
    end 
    mean_noisy_spectral(k) = mean(Acc_noisy_spectral);
    std_noisy_spectral(k) = std(Acc_noisy_spectral);
    mean_noisy_TPT(k) = mean(Acc_noisy_TPT);
    std_noisy_TPT(k) = std(Acc_noisy_TPT);
end