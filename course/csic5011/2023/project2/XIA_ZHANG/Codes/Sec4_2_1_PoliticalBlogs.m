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
plot(G)
c = c([1:148,150:518,520:end]) ;
B = adjacency(G);

% Spectral Clustering & TPT 
[c_sc,Acc_SC] = SpectralClustering(B,c);
% Node source state S(127),  react state S(838）
[c_TPT,J_plus,T,Acc_TPT] = TPT2(B,c,127,838);

% plot the nodes with top15 transition flux.
[M,I] = sort(T,'descend');
ImportantNodes = I(1:15);
OtherNodes = I(16:end);


% effective/transition flux
edges = G.Edges.EndNodes;
[m,~] = size(edges);
s1 = [];
t1 = [];   % J_plus(s1,t1)>0
weight1 = [];

s2 = [];
t2 = [];
for i = 1:m
    x = edges(i,1);
    y = edges(i,2);
    if J_plus(x,y)>0
        s1 = [s1 x];
        t1 = [t1 y];
        weight1 = [weight1, J_plus(x,y)];
    else
        s2 = [s2 x];
        t2 = [t2 y];
    end
end

weight2 = min(J_plus(J_plus>0))*ones(1,length(s2));
di_G = digraph([s1,s2],[t1,t2],[weight1,weight2]);

H = subgraph(di_G,ImportantNodes);
H.Nodes.Name = string(ImportantNodes);

LWidths = 5*H.Edges.Weight/max(H.Edges.Weight);

MSizes = 20*(T(ImportantNodes)/max(T(ImportantNodes)))+5;
figure
P = plot(H,'layout','force','LineWidth',LWidths,'EdgeColor','k');

Liberal = ImportantNodes(c_TPT(ImportantNodes)==0);
Conservative = ImportantNodes(c_TPT(ImportantNodes)==1);
highlight(P, string(Liberal), 'NodeColor','r');
highlight(P, string(Conservative), 'NodeColor','b');
%highlight(P, s2,t2, 'EdgeColor','w');
for i = 1:15
    highlight(P,i,'MarkerSize',MSizes(i))
end

set(gca, 'LooseInset', [0.01,0.01,0.01,0.01]);
saveas(gcf,'Blog_TPT.jpg')



