%把头像和姿态坐标画在平面上

function showFacesOnR2(images,poses,ks)

%normalize into 1:1

poses(1,:)=poses(1,:)/range(poses(1,:));

poses(2,:)=poses(2,:)/range(poses(2,:));

%draw all points

scatter(poses(1,:),poses(2,:),12,'o','filled');

xlabel('left-right pose');

ylabel('up-down pose'); 

hold on

%draw selected points

scatter(poses(1,ks),poses(2,ks),24,'ko');

hold on

%draw images on selected points

scale = 0.001;

x=zeros(64,64);

for p=1:size(ks,2)

    k=ks(p);

    for i=1:64

        x(:,i)=images((i-1)*64+1:i*64,k);

    end

    xc=poses(1,k);

    yc=poses(2,k);

    imshow(xc:scale:xc+64*scale,yc:-scale:yc-64*scale,x);

    hold on

end

return



