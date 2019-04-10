A = ["murder";"rape";"robbery";"assulat";"burglary";"larceny";"auto"];
type = char(A);
B=[    " albuquerque ";    "atlanta";	"austin"	;    "baltimore"	  ;  "boston"	;    "buffalo"	;    "charlotte"	;    "chicago"	;    "cleveland"	 ;   "columbus"	  ;  "dallas"	;    "denver";	    "detroit";	    "elpaso"	;    "fortworth"	;    "fresno"	;    "honolulu"	;    "houston"	;    "indianapolis"	;    "jacksonville";	    "kansascity";	    "losangeles"	  ;  "memphis"	;    "miami";	    "milwaukee"	;    "minneapolis";	    "nashville";	    "neworleans"	   ; "newyork"	;    "oakland";	    "oklacity";	    "omaha";	    "philadelphia";	    "pheonix";	    "pittsburgh"	 ;   "portland";	    "sacramento"	;    "saintlouis"	 ;   "sanantonio"	  ;  "sandiego";	    "sanfran"	;    "sanjose";	    "seattle";	    "toledo"	;    "tucson"	;    "tulsa";	    "washington"	  ;  "birmingham";	    "mesa"	;    "anaheim"	 ;   "saintpeters"	 ;   "tampa"	;    "louisville";	    "saintpaul";	    "jerseycity";	    "newark";	    "akron";	    "arlington";	    "corpuschri"];
cityname=char(B);
filename = 'crime1985.dat';
delimiterIn = ' ';
crime = importdata(filename,delimiterIn);
figure(1);
h=boxplot(crime,type);
set(gca,'FontSize',20)
set(h,'LineWidth',2);
title('Crime data in1985');
xlabel('type','Fontname', 'Times New Roman','FontSize',20);
ylabel('Crime rate','Fontname', 'Times New Roman','FontSize',20);
%correlation of pairwise matrix
M = corr(crime,crime);
% principal components
w = 1./var(crime);
[wcoeff,score,latent,tsquared,explained] = pca(crime,...
'VariableWeights',w);
figure(2);
 plot(latent ,'-*','LineWidth',2);
 set(gca,'FontSize',20)
 title('Eigen values of covariance matrix');
 xlabel('Component number','Fontname', 'Times New Roman','FontSize',20);
ylabel('Eigenvalue','Fontname', 'Times New Roman','FontSize',20);
M3 = wcoeff(:,1:3);
 coefforth = inv(diag(std(crime)))*wcoeff;
 %orthogonal matrix
 I = coefforth'*coefforth;
cscores = zscore(crime)*coefforth;
figure(3)
plot(score(:,1),score(:,2),'*','color','r','MarkerSize',12);
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
set(gca,'FontSize',20)


figure(4)
plot(score(:,1),score(:,3),'o','color','b','MarkerSize',12)
xlabel('1st Principal Component')
ylabel('3nd Principal Component')
set(gca,'FontSize',20)

figure(5)
plot(score(:,2),score(:,3),'+','color','m','MarkerSize',12)
xlabel('2st Principal Component')
ylabel('3nd Principal Component')
set(gca,'FontSize',20) 

figure(6)
biplot(coefforth(:,1:2),'Scores',score(:,1:2),'Varlabels',type);
set(gca,'FontSize',20);
axis([-0.6 0.8 -0.6 0.8]);

figure(7)
biplot(coefforth(:,1:3),'Scores',score(:,1:3),'Varlabels',type);
axis([-.6 0.8 -.6 .8 -1 0.6]);
view([30 40]);
set(gca,'FontSize',20);


