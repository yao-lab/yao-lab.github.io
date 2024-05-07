% CSIC5011: Hw-6, Q3(c) Landmark ISOMAP on
%SWISS ROLL DATASET

  N=2000;
  K=12;
  d=2; 

clf; colordef none; colormap jet; set(gcf,'Position',[200,400,620,200]);

% PLOT TRUE MANIFOLD
  tt0 = (3*pi/2)*(1+2*[0:0.02:1]); hh = [0:0.125:1]*30;
  xx = (tt0.*cos(tt0))'*ones(size(hh));
  yy = ones(size(tt0))'*hh;
  zz = (tt0.*sin(tt0))'*ones(size(hh));
  cc = tt0'*ones(size(hh));

  subplot(1,3,1); cla;
  surf(xx,yy,zz,cc);
  view([12 20]); grid off; axis off; hold on;
  lnx=-5*[3,3,3;3,-4,3]; lny=[0,0,0;32,0,0]; lnz=-5*[3,3,3;3,3,-3];
  lnh=line(lnx,lny,lnz);
  set(lnh,'Color',[1,1,1],'LineWidth',2,'LineStyle','-','Clipping','off');
  axis([-15,20,0,32,-15,15]);

% GENERATE SAMPLED DATA
  tt = (3*pi/2)*(1+2*rand(1,N));  height = 21*rand(1,N);
  X = [tt.*cos(tt); height; tt.*sin(tt)];

% SCATTERPLOT OF SAMPLED DATA
  subplot(1,3,2); cla;
  scatter3(X(1,:),X(2,:),X(3,:),12,tt,'+');
  view([12 20]); grid off; axis off; hold on;
  lnh=line(lnx,lny,lnz);
  set(lnh,'Color',[1,1,1],'LineWidth',2,'LineStyle','-','Clipping','off');
  axis([-15,20,0,32,-15,15]); drawnow;

% Landmark ISOMAP with random k-landmarks ALGORITHM
n=size(X,2);
D=zeros(n,n);
X=X';

k=1000;
idx=randperm(n);
X1=X(idx(1:k),:);
X2=X(idx(k+1:end),:);

A=X1*X1';

[V1,D1]=eig(A);
d1=diag(D1);
[eval,idx1]=sort(d1,'descend');
evec=V1(:,idx1);

X1_new=evec(:,1:2)*sqrtm(diag(eval(1:2)));
B=X1*X2';
X2_new=(B'*evec(:,1:2))/(sqrtm(diag(eval(1:2))));
Y=[X1_new; X2_new];

tt1=tt(1,idx);
% SCATTERPLOT OF EMBEDDING
  subplot(1,3,3); cla;
  scatter(Y(:,1),Y(:,2),12,tt1,'+');
  grid off;
  set(gca,'XTick',[]); set(gca,'YTick',[]); 


