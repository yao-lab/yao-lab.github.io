
repeat=1;
res=[];
for r = [0.1]
    count=0;
    for i = 1:repeat
        count=problem1d(r,1)+count;
    end
    res=[res,count/repeat];
end

