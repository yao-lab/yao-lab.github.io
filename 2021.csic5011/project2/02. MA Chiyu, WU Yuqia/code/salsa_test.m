% webpage = univ_cn;
% L = sign(W_cn);
L = [0 0 1 0 1 0; 1 0 0 0 0 0; 0 0 0 0 1 0; 0 0 0 0 0 0; 0 0 1 1 0 0; 0 0 0 0 1 0];
index = [1, 2, 3, 5, 6, 10];
L_sum1 = sum(L, 1); %row
L_sum2 = sum(L, 2); %column
sizeL = size(L, 1);
Lr = zeros(sizeL, sizeL); Lc = zeros(sizeL, sizeL);

% scale

J1 = find(abs(L_sum1)>0.1); % authority set
J2 = find(abs(L_sum2)>0.1); % hub set

size_J1 = length(J1); size_J2 = length(J2);

for i = 1:size_J1
    Lc(:, J1(i)) = L(:, J1(i))/L_sum1(J1(i));
end
for i = 1:size_J2
    Lr(J2(i), :) = L(J2(i), :)/L_sum2(J2(i));
end
% compute authority and hub matrix

H_tmp = Lr * Lc';
H = H_tmp(J2, J2); % hub matrix

A_tmp = Lc' * Lr;
A = A_tmp(J1, J1); % authority matrix

% find irreducible components of hub
sizeH = size(H, 1);

Hgroup = []; len_groupH = [];
Hgroup1 = find(H(1, :) > 0);
size_tmp =length(Hgroup1);
Hgroup(1, 1:size_tmp) = Hgroup1;
len_groupH(1) = size_tmp;
Hgroup_all = Hgroup1;
t = 1;
for i = 2:sizeH
    if length(Hgroup_all) == sizeH
        break
    end  
    group_tmp = find(H(i, :)> 0);
    Hgroup_sum = sum(Hgroup, 2); 
    Hgroup_size = sum(sign(Hgroup_sum)); % number of existing irreducible components
    for j = 1:Hgroup_size
        if ~all(ismember(group_tmp, Hgroup(j, :)))  
            if ~isempty(intersect(group_tmp, Hgroup(j, :)))  % belong to the same irreducible component
                group_tmp = union(group_tmp, Hgroup(j, :));
                size_tmp = length(group_tmp);
                Hgroup(j, 1:size_tmp) = group_tmp;
                Hgroup_all = union(group_tmp, Hgroup_all);
                len_groupH(j) = size_tmp;
            else
                size_tmp = length(group_tmp);
                Hgroup(Hgroup_size+t, 1:size_tmp) = group_tmp;
                t = t + 1;
                Hgroup_all =  union(group_tmp, Hgroup_all);
            end
        end
    end
end



% 
% % find irreducible component of authority
sizeA = size(A, 1);

Agroup = []; len_groupA = [];
Agroup1 = find(A(1, :) > 0);
size_tmp =length(Agroup1);
Agroup(1, 1:size_tmp) = Agroup1;
len_groupA(1) = size_tmp;
Agroup_all = Agroup1;
t = 1;
for i = 2:sizeA
    if length(Agroup_all) == sizeA
        break
    end  
    group_tmp = find(A(i, :)> 0);
    Agroup_sum = sum(Agroup, 2); 
    Agroup_size = sum(sign(Agroup_sum)); % number of existing irreducible components
    for j = 1:Agroup_size
        if ~all(ismember(group_tmp, Agroup(j, :)))  
            if ~isempty(intersect(group_tmp, Agroup(j, :)))  % belong to the same irreducible component
                group_tmp = union(group_tmp, Agroup(j, :));
                size_tmp = length(group_tmp);
                Agroup(j, 1:size_tmp) = group_tmp;
                Agroup_all = union(group_tmp, Agroup_all);
                len_groupA(j) = size_tmp;
            else
                size_tmp = length(group_tmp);
                Agroup(Agroup_size+t, 1:size_tmp) = group_tmp;
                t = t + 1;
                Agroup_all =  union(group_tmp, Agroup_all);
            end
        end
    end
end
% Authority rank
Ascore = [];
Agroup_num = size(Agroup, 1);
for i = 1:Agroup_num
    group_tmp = setdiff(Agroup(i, :), 0);
    size_tmp = length(group_tmp);
    group_mat = A(group_tmp, group_tmp);
    sum_node = sum(sum(sign(L(:, J1(group_tmp)))));
    for j = 1:size_tmp
        sum_j_node = sum(sign(L(:, J1(group_tmp(j)))));
        Ascore(group_tmp(j)) = size_tmp/size_J1 * (sum_j_node/sum_node);
    end
end
    
[~,id] = sort(Ascore,'descend');
index(J1(id))

% Hub score
Hscore = [];
Hgroup_num = size(Hgroup, 1);
for i = 1:Hgroup_num
    group_tmp = setdiff(Hgroup(i, :), 0);
    size_tmp = length(group_tmp);
    group_mat = H(group_tmp, group_tmp);
    sum_node = sum(sum(sign(L(J2(group_tmp),:))));
    for j = 1:size_tmp
        sum_j_node = sum(sign(L(J2(group_tmp(j)),:)));
        Hscore(group_tmp(j)) = size_tmp/size_J2 * (sum_j_node/sum_node);
    end
end

[~,id] = sort(Hscore,'descend');
index(J2(id))