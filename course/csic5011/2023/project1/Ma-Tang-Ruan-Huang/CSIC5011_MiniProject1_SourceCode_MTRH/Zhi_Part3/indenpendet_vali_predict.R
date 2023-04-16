library( matrixStats )
library(ggplot2)
library(future)
library(dplyr)
set.seed(50)
##########load data##########
work_dir='/Users/joeyhuang/Downloads/share/part3code/'
load(paste(work_dir,'independent_dataset.RData',sep=''))

########### PCA #####
library(RSpectra)
source(paste(work_dir,'RSpectra_pca.R',sep=''))
pca <- prcomp_svds(data_all,k=16)
rownames(info_all)<-info_all$ID

pca.comb.df = cbind(as.data.frame(pca$rotation), info_all)
pca.comb.df$dataset=c(rep('orgin',1043),rep('independent_vali',400))
ggplot(pca.comb.df, aes(PC1, PC2, color = factor(dataset))) + geom_point()+ ggtitle('PCA') + theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
                                                                                                   legend.title = element_blank(), 
                                                                                                   legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
                                                                                                   legend.position='bottom',legend.text=element_text(size=16,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
                                                                                                   axis.text.x=element_text(size=,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
                                                                                                   axis.title.x=element_text(size=23,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=18,face='plain',color='black'))
#ggsave('PCA.png',width = 800, height = 600, units = 'px', dpi = 180)

########### t-sne #####
library(Rtsne)
tsne_result <- Rtsne(pca$rotation)
tsne.comb.df = cbind(as.data.frame(tsne_result$Y), info_all)
tsne.comb.df$dataset=c(rep('orgin',1043),rep('independent_vali',400))
ggplot(tsne.comb.df, aes(V1, V2, color = factor(region))) + geom_point() + ggtitle('t-SNE') + theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
                                                                                                    legend.title = element_blank(), 
                                                                                                    legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
                                                                                                    legend.position='bottom',legend.text=element_text(size=16,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
                                                                                                    axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
                                                                                                    axis.title.x=element_text(size=18,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=18,face='plain',color='black'))
#ggsave('t-SNE.png',width = 800, height = 600, units = 'px', dpi = 180)


######kmeans prediction########
library(cluster)
x_train=cbind(tsne.comb.df$V1,tsne.comb.df$V2)
y_test=pca.comb.df$region[1043:1443]
km=kmeans(x_train,7)
result_kmn=km$cluster

library('mclust')
##extra data locates after the 1043
adjustedRandIndex(result_kmn[1043:1443],y_test)
