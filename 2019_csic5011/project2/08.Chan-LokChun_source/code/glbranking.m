function rank = glbranking(score)
%this function returns the global consented rank from the data based on the
%scores from Batch_Hodgerank
rank = [];
for i=1:4
    [b,idx] = sort(score(:,i));
    rank = [rank idx];
end
end

