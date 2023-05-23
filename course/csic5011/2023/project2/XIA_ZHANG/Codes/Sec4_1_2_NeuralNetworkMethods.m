clear all
clc

%% Draw the graph for data-driven methods
load('karate.mat')
G = graph(A);      % create a graph from edges

% Zachary's Karate Club Dataset
load('Karate Club_Deep Walk_97.06.mat')
coach = find(label==label(1));
figure
P = plot(G); 
set(gca, 'LooseInset', [0.01,0.01,0.01,0.01]);
highlight(P, coach, 'NodeColor','r' ,'EdgeColor', 'r'); % highlight group
saveas(gcf,'Club_DeepWalk.jpg')

%title("Deep Walk")


load('Karate Club_Node2Vec_97.06.mat')
coach = find(label==label(1));
figure
P = plot(G);  
highlight(P, coach, 'NodeColor','r' ,'EdgeColor', 'r'); % highlight group
%title("Node2Vec")
set(gca, 'LooseInset', [0.01,0.01,0.01,0.01]);
saveas(gcf,'Club_Node2Vec.jpg')
