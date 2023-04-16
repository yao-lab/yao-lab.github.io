library(BiocManager)
library( matrixStats )
library(ggplot2)
library(future)
plan(multicore, workers=10)
#setwd(r"(/Users/mrc/Documents/MRC/csic5011/mini_project/part1)")

########## read selected data ##########
data = readRDS('./Yuyan/share/selected.snps.RData')
info = read.csv('/Users/mrc/Documents/MRC/csic5011/mini_project/ceph_hgdp_minor_code_XNA.sampleInformation.csv')
sum(rownames(data) != info$ID)
# Color the points based on population
# > colnames(info)
# [1] "ID"                "Gender"            "Population"       
# [4] "Geographic.origin" "Geographic.area"   "region"           
# [7] "distance"          "latitude"          "longtitude" 

######### pca ######
pca <- prcomp(data)
pca.comb.df = cbind(as.data.frame(pca$x), info)
ggplot(pca.comb.df, aes(PC1, PC2, color = region)) + geom_point()+ ggtitle('PCA') + theme(axis.title = element_blank())
#ggsave('PCA.png',width = 800, height = 600, units = 'px', dpi = 180)
clust <- kmeans(pca$x[,1:2], centers=7)$cluster
clust
write.csv(clust, "pca_cluster.csv")
########### MDS ##########
mds_result <- cmdscale(dist(data))
mds.comb.df = cbind(as.data.frame(mds_result), info)
ggplot(mds.comb.df, aes(V1, V2, color = region)) + geom_point() + ggtitle('MDS') + theme(axis.title = element_blank())
#ggsave('MDS.png',width = 800, height = 600, units = 'px', dpi = 180)
mds.df <- as.data.frame(mds_result)
c("Dim.1", "Dim.2")
kmclusters_mds <- kmeans(mds.df, 7)
kmclusters_mds <- as.factor(kmclusters$cluster) 
mds.df$groups <- kmclusters_mds
mds.df
write.csv(mds.df, "mds_cluster.csv")
########### t-sne #####
library(Rtsne)
tsne_result <- Rtsne(data)
tsne.comb.df = cbind(as.data.frame(tsne_result$Y), info)
ggplot(tsne.comb.df, aes(V1, V2, color = region)) + geom_point() + ggtitle('t-SNE') + theme(axis.title = element_blank())
#ggsave('t-SNE.png',width = 800, height = 600, units = 'px', dpi = 180)
clust_tsne <- kmeans(tsne_result$Y[,1:2], centers=7)$cluster
clust_tsne
write.csv(clust_tsne, "tsne_cluster.csv")
######### isomap ########
#BiocManager::install("RDRToolbox")
library(RDRToolbox)

k = 5 # 10 neighbors
isomap_result <- Isomap(data, dims = 2, k, mod = FALSE, plotResiduals = FALSE, verbose = TRUE)
isomap.comb.df = cbind(as.data.frame(isomap_result$dim2), info)
ggplot(isomap.comb.df, aes(V1, V2, color = region)) + geom_point() + ggtitle(sprintf('ISOMAP (k=%d)',k)) + theme(axis.title = element_blank())
#ggsave('Isomap.png',width = 800, height = 600, units = 'px', dpi = 180)

clust_isomp <- kmeans(isomap_result$dim2[,1:2], centers=7)$cluster
clust_isomp
write.csv(clust_isomp, "isomap_cluster.csv")








####### LLE #########
# library(RDRToolbox)
lle_results = LLE(data, dim=2, 5)
lle.comb.df = cbind(as.data.frame(lle_results), info)
ggplot(lle.comb.df, aes(V1, V2, color = region)) + geom_point() + ggtitle(sprintf('LLE (k=%d)',k)) + theme(axis.title = element_blank())
ggsave('LLE.png',width = 800, height = 600, units = 'px', dpi = 180)

## seems not good
