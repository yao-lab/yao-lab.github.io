rm(list = ls())
library(data.table)
library(dplyr)
library(Rdimtools)
library(RandPro)
library(tibble)
library(ggplot2)
library(ConsensusClusterPlus)
library(ComplexHeatmap)
library(circlize)
setwd("/media/london_A/kewei/MATH5473/")
whitePurple = c("9"='#f7fcfd',"6"='#e0ecf4',"8"='#bfd3e6',"5"='#9ebcda',"2"='#8c96c6',"4"='#8c6bb1',"7"='#88419d',"3"='#810f7c',"1"='#4d004b')

# Load mutation data
mut.df <- fread("./data/ceph_hgdp_minor_code_XNA.betterAnnotated.csv")
mut.df <- mut.df %>% dplyr::select(-c(chr, pos))
mut.df <- column_to_rownames(mut.df, var = "snp")
mut.df <- as.matrix(mut.df)

# Load meta data
meta.info <- read.csv("data/ceph_hgdp_minor_code_XNA.sampleInformation.csv")

#######################
# Perform random projection
set.seed(1234)
rp.res <- do.rndproj(t(mut.df), ndim = dimension(ncol(mut.df), epsilon = 0.2), preprocess = "null", type = "sparse")
saveRDS(rp.res, "RandomProjection.res")

# rp.res.axis <- as.data.frame(rp.res$Y)
# rp.res.axis$region <- meta.info$region
# pdf("rp.pdf", width = 4.5, height = 4)
# ggplot(rp.res.axis, aes(V1, V2, color = region))+
#   geom_point(size = 1.5)+
#   theme_dr()+
#   theme(panel.grid = element_blank(),
#         axis.title = element_text(size = 15),
#         legend.title = element_text(size = 13),
#         legend.text = element_text(size = 12))
# dev.off()

# Perform PCA based on random projection
pca.rp.res <- prcomp(rp.res$Y, center = T, scale = T)
optimal.pc <- findPC::findPC(pca.rp.res$sdev, number = 50, method = "second derivative", figure = T)
pca.axis <- as.data.frame(pca.rp.res$x)
pca.axis$region <- meta.info$region

pdf("pca.rp.pdf", width = 4.5, height = 4)
ggplot(pca.axis, aes(PC1, PC2, color = region))+
  geom_point(size = 1.5)+
  theme_dr()+
  theme(panel.grid = element_blank(),
        axis.title = element_text(size = 15),
        legend.title = element_text(size = 13),
        legend.text = element_text(size = 12))
dev.off()


#############################################
# Clustering
consensus.res <- ConsensusClusterPlus(d = t(pca.rp.res$x[,1:optimal.pc]),
                                      maxK = 7, 
                                      clusterAlg = 'km', 
                                      reps = 100, 
                                      title = "consensus.pca.rp",
                                      seed = 1234,
                                      plot = "png")
cluster <- as.data.frame(consensus.res[[4]][["consensusClass"]])
colnames(cluster) <- "Cluster"; cluster$Cluster <- paste0("C", cluster$Cluster)
saveRDS(cluster, "cluster.rds")
cluster1 <- cluster[order(cluster$Cluster),,drop = F]

consensus.mat <- as.data.frame(consensus.res[[4]][["consensusMatrix"]])
colnames(consensus.mat) <- names(consensus.res[[4]][["consensusClass"]])
rownames(consensus.mat) <- names(consensus.res[[4]][["consensusClass"]])
consensus.mat <- consensus.mat[rownames(cluster1), rownames(cluster1)]
split <- data.frame(c(rep("C1", table(cluster1$Cluster)[1]),
                      rep("C2", table(cluster1$Cluster)[2]),
                      rep("C3", table(cluster1$Cluster)[3]),
                      rep("C4", table(cluster1$Cluster)[4]))
)

ann.col <- c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF")
names(ann.col) <- unique(cluster1$Cluster)
top_ha <- HeatmapAnnotation(Cluster = cluster1$Cluster,
                            col = list(Cluster = ann.col),
                            show_annotation_name = F, 
                            show_legend = F)
row_ha <- rowAnnotation(Cluster = cluster1$Cluster,
                        col = list(Cluster = ann.col),
                        show_annotation_name = F, 
                        show_legend = F)
p <- Heatmap(as.matrix(consensus.mat), 
             col = colorRamp2(seq(0, 1, length.out = length(whitePurple)), whitePurple),
             top_annotation = top_ha,
             right_annotation = row_ha,
             cluster_columns = F, 
             cluster_rows = F, 
             show_row_names = F, 
             show_column_names = F, 
             column_split = split,
             column_title = "Sample clusters",
             row_split = split,
             row_title_rot = 0, 
             row_title_side = "right",
             row_gap = unit(0, "mm"), 
             column_gap = unit(0, "mm"),
             border = T,use_raster = T,
             heatmap_legend_param = list(
               title = "consensus",
               # at = c(-0.1, 0, 0.1),
               direction = "horizontal"
               # labels = c("-1", "0", "1")
             )
)
pdf("Sampleluster.consensusMat.pdf", width = 5.2, height = 5.2)
draw(p, heatmap_legend_side="bottom")
dev.off()

pca.axis$Cluster <- cluster$Cluster
pdf("pca.rp.cluster.pdf", width = 4.5, height = 4)
ggplot(pca.axis, aes(PC1, PC2, color = Cluster))+
  geom_point(size = 1.5)+
  theme_dr()+
  theme(panel.grid = element_blank(),
        axis.title = element_text(size = 15),
        legend.title = element_text(size = 13),
        legend.text = element_text(size = 12))
dev.off()