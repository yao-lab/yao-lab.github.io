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

% Plot the results
G = graph(A);      % create a graph

% Bisection
figure
P = plot(G);  
highlight(P, coach_equal, 'NodeColor','r' ,'EdgeColor', 'r'); % highlight group
title("Spectral Clsutering + Bisection")

% TPT
figure
P = plot(G);  
highlight(P, coach_TPT, 'NodeColor','r' ,'EdgeColor', 'r'); % highlight group
title("Transition Path")


