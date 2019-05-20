function [data_trans] = age_trans(agedata)
% transform nx4 agedata matrix into nx2 data matrix which is compatible
% with Batch_Hodgerank
m = size(agedata,1);
data_trans = zeros(m,2);
for i=1:m
    if agedata(i,4) == -1
        data_trans(i,1) = agedata(i,3);
        data_trans(i,2) = agedata(i,2);
    else 
        data_trans(i,1) = agedata(i,2);
        data_trans(i,2) = agedata(i,3);
    end
end

