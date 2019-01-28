% READONEQUERY
%
%   numClick: number of clicks per record
%   txt: the text part of data 
%   raw: raw data
[numClick,txt,raw]=xlsread('logdata-onequery.xls');


% qid: query id, here 0 or 1
numRecords = size(raw,1);
% for i=1:numRecords,
%     qid(i)=str2num(txt{i+1,1}(2));
% end

% uid: url id
uid=[];
for i=1:numRecords,
    I=regexp(raw{i,3},'"u');
    J=regexp(raw{i,3},'\d"');
    if length(I)~=length(J),
        error('Wrong Length!');
    end,
    for j=1:length(I),
        uid=union(uid,str2num(raw{i,3}(I(j)+2:J(j))));
    end,
end

% cid: click id per record
% urls: urls per record, two pages in maximal for each record
cid=cell(numRecords,5);
urls = cell(numRecords,5);

for i=1:numRecords,
    idPage = sscanf(raw{i,2}(2:end-1),'%d,');
    numPages(i) = length(idPage);
    I=regexp(raw{i,4},'\],\[');
    if ~isempty(I),
        str =[];
        clicks = [];
        utxt = [];
        for j=1:numPages(i),
            if j==1,
                startID=3;
                endID = I(j)-1;
            end
            if j>1 & j<numPages(i),
                startID=I(j-1)+3;
                endID = I(j)-1;
            end
            if j==numPages(i),
                startID=I(j-1)+3;
                endID = length(raw{i,4})-2;
            end
            
            str{j}=raw{i,4}(startID:endID);
            clicks{j}=sscanf(str{j},'%d,');
            
            J=regexp(raw{i,3},'\],\[');
            if j==1,
                startID=3;
                endID = J(j)-1;
            end
            if j>1 & j<numPages(i),
                startID=J(j-1)+3;
                endID = J(j)-1;
            end    
            if j==numPages(i),
                startID=J(j-1)+3;
                endID = length(raw{i,3})-2;
            end
            utxt{j} = raw{i,3}(startID:endID);
            urls{i,j} = sscanf(utxt{j},'"u%d",');
            cid{i,j}=urls{i,j}(clicks{j});
        end
        
    else
        str = [];
        clicks = [];
        utxt = [];
        str = raw{i,4}(3:end-2);
        clicks = sscanf(str,'%d,');
        utxt = raw{i,3}(3:end-2);        
        urls{i,1} = sscanf(utxt,'"u%d",');
        cid{i,1} = urls{i,1}(clicks);
    end
end

% p_click: probability distribution of click frequency
p_click=zeros(length(uid),1);
for i=1:numRecords,
    for j=1:size(cid,2),
        p_click(cid{i,j})=p_click(cid{i,j})+numClick(i);
    end
end

% T_click:  an example of pairwise comparison for each click pattern, 
%           replace this with your personal implementations
T_click = zeros(length(uid),length(uid));
N_click = T_click;
TT_click = T_click;
NN_click = TT_click;
for i=10:10,%numRecords,
    IDskip_base = [];
    for j=1:size(cid,2),
        if j>1,
            IDskip_base = setdiff(union(urls{i,j-1},IDskip_base),cid{i,j-1});
        end
        for k=1:length(cid{i,j}),
            if k>0,
                jd = find(urls{i,j}==cid{i,j}(k));
                
                IDskip = union(IDskip_base,setdiff(urls{i,j}(1:jd-1),cid{i,j}(1:k-1)));
  
                % T_click: the first model which only counts the skipped urls
                T_click(cid{i,j}(k),IDskip)=T_click(cid{i,j}(k),IDskip)+ numClick(i);
                N_click(cid{i,j}(k),IDskip)=N_click(cid{i,j}(k),IDskip)+ numClick(i);
                N_click(IDskip,cid{i,j}(k))=N_click(cid{i,j}(k),IDskip);
                
                % TT_click: the second model which counts the second url as
                % skipped when the first url is clicked.
                TT_click(cid{i,j}(k),IDskip)=TT_click(cid{i,j}(k),IDskip)+ numClick(i);
                NN_click(cid{i,j}(k),IDskip)=NN_click(cid{i,j}(k),IDskip)+ numClick(i);
                NN_click(IDskip,cid{i,j}(k))=NN_click(cid{i,j}(k),IDskip);
                if jd==1 & j==1,
                    TT_click(urls{i,j}(jd),urls{i,j}(jd+1))=TT_click(urls{i,j}(jd),urls{i,j}(jd+1))+ numClick(i);
                    NN_click(urls{i,j}(jd),urls{i,j}(jd+1))=NN_click(urls{i,j}(jd),urls{i,j}(jd+1))+ numClick(i);
                    NN_click(urls{i,j}(jd+1),urls{i,j}(jd))=NN_click(urls{i,j}(jd),urls{i,j}(jd+1));
                end                
            end
        end
    end
end
