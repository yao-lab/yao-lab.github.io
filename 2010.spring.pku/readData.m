% READDATA
%
%   numClick: number of clicks per record
%   txt: the text part of data 
%   raw: raw data
[numClick,txt,raw]=xlsread('CLickLogSample.xls');


% qid: query id, here 0 or 1
numRecords = size(txt,1)-1;
for i=1:numRecords,
    qid(i)=str2num(txt{i+1,1}(2));
end

% uid: url id
uid=[];
for i=1:numRecords,
    I=regexp(txt{i+1,2},'"u');
    J=regexp(txt{i+1,2},'\d"');
    if length(I)~=length(J),
        error('Wrong Length!');
    end,
    for j=1:length(I),
        uid=union(uid,str2num(txt{i+1,2}(I(j)+2:J(j))));
    end,
end

% cid: click id per record
% urls: urls per record, two pages in maximal for each record
cid=cell(numRecords,2);
urls = cell(numRecords,2);

for i=1:numRecords,
    I=regexp(txt{i+1,3},'\],\[');
    if ~isempty(I),
        str =[];
        clicks = [];
        str{1}=txt{i+1,3}(3:I(1)-1);
        str{2}=txt{i+1,3}(I(1)+3:end-2);
        clicks{1}=sscanf(str{1},'%d,');
        clicks{2}=sscanf(str{2},'%d,');
        J=regexp(txt{i+1,2},'\],\[');
        utxt = [];
        utxt{1} = txt{i+1,2}(3:J(1)-1);
        urls{i,1} = sscanf(utxt{1},'"u%d",');
        utxt{2} = txt{i+1,2}(J(1)+3:end-2);
        urls{i,2} = sscanf(utxt{2},'"u%d",');
        cid{i,1}=urls{i,1}(clicks{1});
        cid{i,2}=urls{i,2}(clicks{2});
    else
        str = [];
        clicks = [];
        utxt = [];
        str = txt{i+1,3}(3:end-2);
        clicks = sscanf(str,'%d,');
        utxt = txt{i+1,2}(3:end-2);        
        urls{i,1} = sscanf(utxt,'"u%d",');
        cid{i,1} = urls{i,1}(clicks);
    end
end

% p_click: probability distribution of click frequency
p_click=zeros(length(uid),1);
for i=1:sum(qid==1);% numRecords,
    for j=1:2,
        p_click(cid{i,j})=p_click(cid{i,j})+numClick(i);
    end
end

% T_click:  an example of pairwise comparison for each click pattern, 
%           replace this with your personal implementations
T_click = zeros(length(uid),length(uid));
for i=1:sum(qid==1),
    for j=1:2,
        for k=1:length(cid{i,j}),
            if k>0,
                jd = find(urls{i,j}==cid{i,j}(k));
                %if ~isempty(jd),
                T_click(cid{i,j}(k),setdiff(urls{i,j}(1:jd),cid{i,j}(1:k-1)))=T_click(cid{i,j}(k),setdiff(urls{i,j}(1:jd),cid{i,j}(1:k-1)))+ numClick(i);
                % T_click(cid{i,j}(k),urls{i,j})=T_click(cid{i,j}(k),urls{i,j})+numClick(i);
            end
        end
    end
end
