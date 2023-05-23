% visualize the dataset
% Club data
clear all
clc
load('karate.mat')

G = graph(A);
coach =find(c0==0);
figure
P = plot(G); 

highlight(P,coach, 'NodeColor','r' ,'EdgeColor', 'r'); % highlight group
set(gca, 'LooseInset', [0.01,0.01,0.01,0.01]);
saveas(gcf,'Club_Data.jpg')

clear all
clc
load('political blogs data.mat')

% Get the maximum undirected, connected graph
[n,~] = size(A);
for i = 1:n
    for j = 1:n
        if A(i,j)~=0
            A(i,j) = 1;
            A(j,i) = 1;
        else
        end
    end
end

S = find(sum(A,2)>0); % connect nodes
c = label(S)';
B = A(S,S);

G = graph(B);
% delete node 519 149
G = rmnode(G,[149,519]);
c = c([1:148,150:518,520:end]) ;

Liberal = find(c==0);
P = plot(G);
edges = G.Edges.EndNodes;
highlight(P,edges(:,1),edges(:,2),'EdgeColor', 'k'); % highlight group
highlight(P,Liberal, 'NodeColor','r'); % highlight group
set(gca, 'LooseInset', [0.01,0.01,0.01,0.01]);
saveas(gcf,'Blog_Data.jpg')