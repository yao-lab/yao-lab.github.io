% Load the pairwise comparison data
load data/incomp.mat

% Move to the code directory
cd code

% Compute HodgeRank scores
[score, totalIncon, harmIncon] = Hodgerank(incomp);

% Display the scores for each video in each model
disp('HodgeRank Scores:')
disp(score)

% Display the percentage of harmonic inconsistency in the total inconsistency
disp(['Percentage of harmonic inconsistency: ' num2str(harmIncon/totalIncon * 100) '%'])

% Load the weblink data
load data/univ_cn.mat

% Construct the pairwise comparison data using the web graph
A = sparse(out_link, in_link, 1, max(out_link), max(in_link));
incomp = [find(A); find(A')]' + [0 size(A,1)];
incomp(:,2) = incomp(:,2) - size(A,1);

% Compute HodgeRank scores
[score, totalIncon, harmIncon] = Hodgerank(incomp);

% Display the scores for each university in each model
disp('HodgeRank Scores:')
disp(score)

% Compare the HodgeRank scores with PageRank and HITs
pr_scores = full(sum(A,2));
pr_scores = pr_scores / sum(pr_scores);
hits_scores = sum(A,1)';
hits_scores = hits_scores / sum(hits_scores);
disp('PageRank Scores:')
disp(pr_scores)
disp('HITs Scores:')
disp(hits_scores)