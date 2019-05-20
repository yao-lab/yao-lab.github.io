clear 
close
load('Agedata.mat')
load('Groundtruth.mat')
%% ground-truth ranking
[B,I] = sort(Age(:,2));
gtranking = I;

%% full-data
compdata = age_trans(Pair_Compar);
[score1,totalIncon1,harmIncon1] = Batch_Hodgerank(compdata, 30);
ranking1 = glbranking(score1);

% accuracy
accuracy1 = [];
for i=1:4
    r = abs(corr(gtranking, ranking1(:,i), 'Type','Spearman'));
    accuracy1 = [accuracy1 r]
end

% plot predicted score by each 4 models
lh = plot([gtranking ranking1]);
set(lh(1), 'Linestyle', '-');
set(lh(2), 'Linestyle', ':');
set(lh(3), 'Linestyle', '-.');
set(lh(4), 'Linestyle', '--');
set(lh(5), 'Linestyle', 'none', 'Marker', 'v')
legend('Ground-truth', 'Uniform', 'B-T', 'T-M', 'Angular transform');
xlabel('Image ID');
ylabel('Ranking');

%%  access each voters' quality
incon_tot = [];
incon_har = [];
for i=1:94
    firstcolumn = Pair_Compar(1,:);
    firstcolumn = Pair_Compar(:,1);
    ith_voter_choice = compdata(firstcolumn == i,:);
    [score,totalIncon, harmIncon] = Batch_Hodgerank(ith_voter_choice, 30);
    incon_tot = [incon_tot totalIncon];
    incon_har = [incon_har harmIncon];
end
incon_tot = mean(incon_tot, 1);
incon_har = mean(incon_har, 1);
stacked = [incon_tot;incon_har];
%% bar plot
bar(stacked')
xlabel('Voters ID');
ylabel('Inconsistency');
legend('incon total', 'incon harmonic')

%% Effect of random selection on data
randsample_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
incon_tot_rand = [];
incon_har_rand = [];
accu_rand = [];

for j=1:length(randsample_rate)
    rand_idx = randsample(14011, ceil(14011*randsample_rate(j)));
    [score,totalIncon, harmIncon] = Batch_Hodgerank(compdata(rand_idx,:), 30);
    incon_tot_rand = [incon_tot_rand totalIncon];
    incon_har_rand = [incon_har_rand harmIncon];
    rank = glbranking(score);
    accuracy = [];
    for i=1:4
        r = abs(corr(gtranking, rank(:,i), 'Type','Spearman'));
        accuracy = [accuracy; r]
    end
    accu_rand = [accu_rand accuracy];
end

%% Plot
subplot(1,3,1)
plot(incon_tot_rand')
legend('Uniform', 'B-T', 'T-M', 'Angular transform');
xlabel('Sampling Rate');
ylabel('Total inconsistency')

subplot(1,3,2)
plot(incon_har_rand')
legend('Uniform', 'B-T', 'T-M', 'Angular transform');
xlabel('Sampling Rate');
ylabel('Harmonic inconsistency')

subplot(1,3,3)
plot(accu_rand')
legend('Uniform', 'B-T', 'T-M', 'Angular transform');
xlabel('Sampling Rate');
ylabel('Accuracy')

%% Removing individuals with high inconsistencies
rmv_incon = ~or(incon_tot > 0.6, incon_har > 0.3);
b = zeros(14011,1);
for k=1:94
    if rmv_incon(k) == 1
        a = (firstcolumn == k);
        b = or(a,b);
    end
end
compdata_new = compdata(b,:);
[score_new,totalIncon_new, harmIncon_new] = Batch_Hodgerank(ith_voter_choice, 30);
ranking_new = glbranking(score_new); % get new predicted global ranking

% get new accuracy
accuracy_new = [];
for i=1:4
    r = abs(corr(gtranking, ranking_new(:,i), 'Type','Spearman'));
    accuracy_new = [accuracy_new r]
end

%% Compare original data and trimmed data
c = categorical({'Uniform', 'B-T', 'T-M', 'Angular transform'});
subplot(1,3,1)
bar(c, [accuracy1; accuracy_new]')
legend('orig model', 'new model');
ylabel('Accuracy (Spearman)')

subplot(1,3,2)
bar(c, [totalIncon1 totalIncon_new])
legend('orig model', 'new model');
ylabel('Total Inconsistency')

subplot(1,3,3)
bar(c, [harmIncon1 harmIncon_new])
legend('orig model', 'new model');
ylabel('Harmonic Inconsistency')