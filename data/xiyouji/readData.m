% This script reads all 100 chapters for the Journal to the West, 
% in file chap001.txt, ..., chap100.txt and returns the following:
%   Y: the whole character-scene occurance matrix
%   name: character names
%   chapter: 100 cells with each cell containing the scenes
%   ID_scene: a map from column to the scene number in chapter
%       ID_scene(1,i): chapter # of the i-th column scene
%       ID_scene(2,i): scene # in chapter of the i-th column scene

% Yuan Yao
%   Peking University
%   Dec 14, 2011

if ~exist('saveData','var'),
    saveData = 1;
end

Y = [];

fnames = dir('chap*.txt');

numFiles = length(fnames);

for K=1:numFiles, 

%    fname = sprintf('chap00%d.txt',K);
    fname = fnames(K).name;
    IDchap = sscanf(fnames(K).name,'chap%d.txt');

    fid = fopen(fname,'r');

    ln = fgetl(fid);
    if ~isempty(ln),

        pos = regexp(ln,'\t');

        numScene = length(pos);
    else
        error('empty line');
    end

    scene = cell(1,numScene);
    for k = 1:numScene,
        if k == numScene,
            scene{k} = ln(pos(k)+1:end);
        else
            scene{k}=ln(pos(k)+1:pos(k+1)-1);
        end
    end

    if ~exist('name','var'),
        lenName = 111;
    else
        lenName = length(name);
    end
    
    X = zeros(lenName,numScene);

    for i = 1:111
        ln = fgetl(fid);
        pos = regexp(ln,'\t');
        name{i} = ln(1:pos(1)-1);
        clean_pos = regexp(name{i},'''');
        for j=length(clean_pos):-1:1,
            name{i}(clean_pos(j))=[];
        end
        for j = 1:numScene,
            if j==numScene,
                x = sscanf(ln(pos(j)+1:end),'%d');
                if isempty(x),
                    X(i,j) = 0;
                else
                    X(i,j) = x(1);
                end
            else
                x = sscanf(ln(pos(j)+1:pos(j+1)-1),'%d');
                if isempty(x),
                    X(i,j) = 0;
                else
                    X(i,j) = x(1);
                end
            end
        end
    end
    
    % Go to additional characters
    ln = fgetl(fid);
    
    while isempty(strmatch(ln,-1)) & ~isempty(deblank(ln)),
        pos = regexp(ln,'\t');
        newname = ln(1:pos(1)-1);
        
        if ~isempty(newname),

            clean_pos = regexp(newname,'''');
            for j=length(clean_pos):-1:1,
                newname(clean_pos(j))=[];
            end

            lenName = length(name);
            nameFound = 0;
            cmp = strmatch(newname,name);
            if ~isempty(cmp),
                rowID = cmp;
            else
                rowID = lenName + 1;
                name{rowID} = newname;
                X(rowID,:) = zeros(1,numScene);
            end

            for j = 1:numScene,
                if j==numScene,
                    x = sscanf(ln(pos(j)+1:end),'%d');
                    if isempty(x),
                        X(rowID,j) = 0;
                    else
                        X(rowID,j) = x(1);
                    end
                else
                    x = sscanf(ln(pos(j)+1:pos(j+1)-1),'%d');
                    if isempty(x),
                        X(rowID,j) = 0;
                    else
                        X(rowID,j) = x(1);
                    end
                end
            end
        end
                
        ln = fgetl(fid);
        rowID = 0;
    end
    
    fclose(fid);
    
    d1 = sum(X,1);
    
    jd = find(d1>0);
    
    realScene = cell(1,length(jd));
    for k=1:length(jd),
        realScene{k} = scene{jd(k)};
    end
    chapter{IDchap}=realScene;
    X = X(:,jd);

    if size(Y,1)~=size(X,1),
        Y = [[Y; zeros(size(X,1)-size(Y,1),size(Y,2))], X];
    else
        Y = [Y, X];
    end

end

if saveData,
    d2 = sum(Y,2);
    id = find(d2>0); 
    
    %%% Remove the following empty rows
    %
    %     name{find(d2==0)}
    % 
    % ans =
    % 
    % 黄花观百眼魔君
    % 
    % 
    % ans =
    % 
    % 竹节山九灵元圣六孙
    % 
    % 
    % ans =
    % 
    % 竹节山九灵元圣
    % 
    % 
    % ans =
    % 
    % 通臂猿猴
    % 
    % 
    % ans =
    % 
    % 麒麟山獬豸洞报事无名小妖
    %
    
    % The last character 大学士萧?奏 is the same as id(130): 萧?    
    Y(id(130),:)=Y(id(130),:)+Y(end,:);
    id(end)=[];
    
    for i=1:length(id),
        nameNew{i}=name{id(i)};
    end
    nameNew{130}='大学士萧?';
    name = nameNew;
        
    
    X = Y(id,:);

    cnt=0;
    ID_scene = [];
    for i=1:100,
        for j=1:length(chapter{i}),
            cnt = cnt + 1;
            ID_scene(1,cnt)=i;
            ID_scene(2,cnt)=j;
        end
    end
    save xiyouji.mat X name chapter ID_scene
end