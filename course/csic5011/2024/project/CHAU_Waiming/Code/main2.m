clear all
close all

load('mnist.mat')
s = RandStream('mlfg6331_64'); 

numTrain=5000;
rep=10;

Xtrain_u = reshape(training.images, [784,60000])';

A=zeros(2,6);

for j=1:rep
    y = randsample(s,60000,numTrain);
    Xtrain=Xtrain_u(y,:);
    Ytrain=training.labels(y);

    for i=1:7
        
        method=i;

        if(method==1)
            [~,train_emb,~] = pca(Xtrain,'NumComponents',2);
        
        elseif(method==2)
            train_emb=LDA( Xtrain , Ytrain);
            train_emb=train_emb(:,1:2);
        
        elseif(method==3)
            [train_emb, ~] = Isomap(Xtrain, 2, 15);
        
        elseif(method==4)
            train_emb= lle(Xtrain',5,2);
            train_emb=train_emb';
    
        elseif(method==5)
            train_emb= laplacianEigenmaps(Xtrain,'NumDimensions',2);
        
        elseif(method==6)
            train_emb= tsne(Xtrain,'NumDimensions',2);
    
        end

        train_emb=rescale(train_emb);
 
        %%%%%%%Between-cluster variation
        m=mean(train_emb);
        B=0;
        for k=0:9

            idk=find(Ytrain==k);

            tmp = train_emb(idk,:);

            mk=mean(tmp);

            B=B+size(idk,1)*norm(mk-m)^2;

        end

         A(1,i)=A(1,i)+B;

         %%%%%%%within-cluster variation
        C=0;
        for k=0:9

            idk=find(Ytrain==k);

            tmp = train_emb(idk,:);
            
            mk=mean(tmp);

            tmp=tmp-mk;

            C=C+norm(tmp,"fro")^2;

        end

        A(2,i)=A(2,i)+C;


    end
end 

A=A./rep;


