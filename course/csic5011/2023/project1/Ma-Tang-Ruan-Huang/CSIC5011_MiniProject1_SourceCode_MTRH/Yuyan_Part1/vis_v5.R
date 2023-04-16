library( matrixStats )
library(ggplot2)
library(future)
plan(multicore, workers=16)
#setwd("/home/yruanaf/storage/snp_proj")
setwd("~/Dropbox/HKUST-courses/CSIC5011/MiniProject1/Yuyan/vis/")
#info = read.csv('datasets/ceph_hgdp_minor_code_XNA.sampleInformation.csv')
info = read.csv('../../data/ceph_hgdp_minor_code_XNA.sampleInformation.csv')
###### PCA #######
pca.df = read.csv(sprintf('pca_all.csv'))
pca.p = ggplot(pca.df, aes(PC1, PC2, color = region)) + 
  geom_point(alpha=.8, show.legend = F)+ ggtitle('PCA') + 
  labs(x = "PC1", y = "PC2") + 
    theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), plot.title = element_text(size = 18, face = 'bold', hjust = 0.5),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=14,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=12,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=12,face='plain',color='black'))
pca.p
ggsave("pca.pdf", width = 5, height = 4.5)
###### ISOMAP ####
k = 10
isomap.df = read.csv(sprintf('ISOMAP (k=%d).all.csv', k))
isomap.p = ggplot(isomap.df, aes(V1, V2, color = region)) +
  geom_point(alpha=.8, show.legend = F)+ ggtitle('ISOMAP') + 
  labs(x = "ISOMAP1", y = "ISOMAP2") + 
  theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), plot.title = element_text(size = 18, face = 'bold', hjust = 0.5),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=14,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=12,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=12,face='plain',color='black'))
isomap.p
ggsave("isomap.pdf", width = 5, height = 4.5)

######### MDS ############
mds.df = read.csv('isomds_minkowski_all.csv')
mds.p = ggplot(mds.df, aes(V1, V2, color = region)) +
  ggtitle('MDS(minkowski)') + 
  geom_point(alpha=.8, show.legend = F)+
  labs(x = "MDS1", y = "MDS2") + 
  theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), plot.title = element_text(size = 18, face = 'bold', hjust = 0.5),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=14,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=12,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=12,face='plain',color='black'))
mds.p
ggsave("mds.pdf", width = 5, height = 4.5)

mds.man.df = read.csv('isomds_manhattan_all.csv')
mds.man.p = ggplot(mds.man.df, aes(V1, V2, color = region)) + geom_point()+ ggtitle('MDS(manhattan)') + 
    theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), 
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=16,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=18,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=18,face='plain',color='black'))
    # theme(legend.position='bottom',legend.justification='center',legend.box='horizontal')


########## UMAP ##########
umap.df = readRDS('UMAP_DR_results.rds')
colnames(umap.df$layout) <- c('V1','V2')
umap.df.comb = cbind(umap.df$layout, info)
umap.p = ggplot(umap.df.comb, aes(V1, V2, color = region)) + 
  geom_point(alpha=.8, show.legend = F)+ ggtitle('UMAP') + 
  labs(x = "UMAP1", y = "UMAP2") + 
  theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), plot.title = element_text(size = 18, face = 'bold', hjust = 0.5),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=14,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=12,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=12,face='plain',color='black'))
umap.p
ggsave("umap.pdf", width = 5, height = 4.5)

########## t-SNE ##########
tsne.df = read.csv('tsne.all.csv')
tsne.p = ggplot(tsne.df, aes(V1, V2, color = region)) + 
  geom_point(alpha=.8, show.legend = F)+ ggtitle('t-SNE') + 
  labs(x = "tSNE1", y = "tSNE2") + 
  theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), plot.title = element_text(size = 18, face = 'bold', hjust = 0.5),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=14,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=12,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=12,face='plain',color='black'))
tsne.p
ggsave("tsne.pdf", width = 5, height = 4.5)
####### LLE ####
lle.df = readRDS('LLE_DR_results.rds')
lle.p = ggplot(lle.df, aes(V1, V2, color = region)) +   
  geom_point(alpha=.8, show.legend = F)+ ggtitle('LLE') + 
  labs(x = "LLE1", y = "LLE2") + 
  theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), plot.title = element_text(size = 18, face = 'bold', hjust = 0.5),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=14,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=12,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=12,face='plain',color='black'))
lle.p
ggsave("lle.pdf", width = 5, height = 4.5)

####### ROSPCA ####
rospca.df = readRDS('ROSPCA_DR_results.rds')
rospca.df.comb = cbind(rospca.df$scores, info)
rospca.p = ggplot(rospca.df.comb, aes(PC1, PC2, color = region)) +   
  geom_point(alpha=.8, show.legend = F)+ ggtitle('Robust PCA') + 
  labs(x = "PC1", y = "PC2") + 
  theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), plot.title = element_text(size = 18, face = 'bold', hjust = 0.5),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=14,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=12,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=12,face='plain',color='black'))
rospca.p
ggsave("rospca.pdf", width = 5, height = 4.5)


#### sammon ####
sammon.df = read.csv('sammon_all.csv')
sammon.p = ggplot(sammon.df, aes(V1, V2, color = region)) + 
  geom_point(alpha=.8, show.legend = T)+ ggtitle('Sammon') + 
  labs(x = "Sammon1", y = "Sommon2") + 
  theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), plot.title = element_text(size = 18, face = 'bold', hjust = 0.5),
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=14,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=12,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=12,face='plain',color='black'))
sammon.p
ggsave("sammon.pdf", width = 5, height = 4.5)

library(patchwork)
library(ggpubr)
# umap.p + lle.p + rospca.p
ggarrange(pca.p,mds.p, rospca.p, umap.p,tsne.p, isomap.p, lle.p, sammon.p, ncol=4, nrow=2, common.legend = TRUE, legend='bottom')+
    theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), 
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=16,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=18,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=18,face='plain',color='black'))
    # theme(legend.position='bottom',legend.justification='center',legend.box='horizontal')
    # theme(legend.position='bottom',legend.justification='center',legend.box='horizontal')
ggsave('comb.png',width = 2600, height = 1200, units = 'px', dpi = 180)
# conda install -c conda-forge r-ggpubr

ggarrange(pca.p, umap.p,tsne.p, isomap.p, lle.p, ncol=5, nrow=1, common.legend = TRUE, legend='bottom')+
    theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), 
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=16,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='bold',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=18,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=18,face='plain',color='black'))
    # theme(legend.position='bottom',legend.justification='center',legend.box='horizontal')
ggsave('comb2.png',width = 2600, height = 1200, units = 'px', dpi = 180)
# conda install -c conda-forge r-ggpubr