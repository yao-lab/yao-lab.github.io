clear all
close all

load('mnist.mat')
s = RandStream('mlfg6331_64'); 
numTrain=5000;
rep=10;

Xtrain_u = reshape(training.images, [784,60000])';



A=zeros(7,2);

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
            train_emb=train_emb(:,1:2);
        
        elseif(method==6)
            train_emb= tsne(Xtrain,'NumDimensions',2);
    
        elseif(method==7)
            train_emb=Xtrain;
        
        end
        

     if(j==1)
        figure
        gscatter(train_emb(:,1),train_emb(:,2),Ytrain);
        axis off
        legend off
     
        filename=strcat('embedd',num2str(method));
        print(gcf,filename,'-djpeg')
     end
    
        
        % figure
        % gscatter(train_emb(k49,1),train_emb(k49,2),Ytrain(k49));
        % axis off
        % legend off
        % 
        % 
        % filename=strcat('embedd49',num2str(method));
        % print(gcf,filename,'-djpeg')
        % 
        %%%%clustering
        idx = kmeans(train_emb,10,'MaxIter',500); 
        
        if(j==1)
            figure
            gscatter(train_emb(:,1),train_emb(:,2),idx);
            axis off
            legend off
    
            filename=strcat('embedd_cluster',num2str(method));
            print(gcf,filename,'-djpeg')
        end
        
        %%%%relabeling the index
        oidx = zeros(size(idx));
        
        for k = 1:10
            idx_k = find(idx== k); 
            label = mode(Ytrain(idx_k));
            oidx(idx_k) = label;  
        end
        
        
        %%%%purity
        purity = 0;
        
        for k=0:9
            idx_k = find(oidx== k); 
            purity = purity + sum(Ytrain(idx_k) == k);
        end
        purity = purity/numTrain;
        A(i,1)=A(i,1)+purity;
        
        %%%%% nmi
        Infor=nmi(idx,Ytrain);
        A(i,2)=A(i,2)+Infor;
    end
end

A=A./rep;



