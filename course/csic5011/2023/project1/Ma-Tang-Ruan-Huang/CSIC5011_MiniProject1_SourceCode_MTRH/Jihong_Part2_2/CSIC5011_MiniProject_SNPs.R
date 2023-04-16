library(tidyverse)

# 1 - load data ---- 
df_SNPs <- read.csv("data/ceph_hgdp_minor_code_XNA.betterAnnotated.csv")
cl <- read.csv("data/ceph_hgdp_minor_code_XNA.sampleInformation.csv")
row.names(df_SNPs) <- df_SNPs$snp
df_SNPs <- df_SNPs[, -seq(1,3)]

mt_SNPs <- t(df_SNPs)

# var <- apply(df_SNPs, 1, var)
# test <- df_SNPs[order(var, decreasing = T)[1:10000], ]

# 2 - Dimension Reduction Methods ----
## LLE ----
library(RDRToolbox)
lle_results <- LLE(mt_SNPs, dim=2, 5)
lle.comb.df <- cbind(as.data.frame(lle_results), cl)
saveRDS(lle.comb.df, "results/LLE_DR_results.rds")
#ggplot(lle.comb.df, aes(V1, V2, color = region)) + geom_point() + ggtitle(sprintf('LLE (k=%d)',5)) + theme(axis.title = element_blank())
#ggsave('LLE.png',width = 800, height = 600, units = 'px', dpi = 180)

## UMAP ---- 
library(umap)
umap.args <- umap(mt_SNPs, random_state=123)
saveRDS(umap.args, "results/UMAP_DR_results.rds")

## robust-PCA ----
library(rospca)
robpca <- robpca(mt_SNPs, 2, 5, ndir = 5000)
saveRDS(robpca, "results/ROSPCA_DR_results.rds")

## random-projection -----
# implemented in python

# 3- clustering ----
# LLE
lle_results <- readRDS("results/LLE_DR_results.rds")
km_lle <- kmeans(lle_results[, 1:2], centers = 7, nstart = 20)
clust_lle <- km_lle$cluster

# UMAP 
umap_results <- readRDS("results/UMAP_DR_results.rds")
km_umap <- kmeans(umap_results$layout, centers = 7, nstart = 20)
clust_umap <- km_umap$cluster

# ROSPCA 
rospca_results <- readRDS("results/ROSPCA_DR_results.rds")
km_rospca <- kmeans(rospca_results$scores, centers = 7, nstart = 20)
clust_rospca <- km_rospca$cluster

# Random Projection
randpro_results <- read.csv("results/RandProj_DR_results.csv")
#sum(randpro_results$X == cl$ID)
km_randpro <- kmeans(randpro_results[, -1], centers = 7, nstart = 20)
clust_randpro <- km_randpro$cluster

# MDS sammon
mds_sammon_results <- read.csv("results/sammon_all.csv")
km_mds_sammon <- kmeans(mds_sammon_results[, c(2, 3)], centers = 7, nstart = 20)
clust_mds_sammon <- km_mds_sammon$cluster

# isoMDS manhattan
isomds_manhattan <- read.csv("results/isomds_manhattan_all.csv") 
km_isomds_manhattan <- kmeans(isomds_manhattan[, c(2, 3)], centers = 7, nstart = 20)
clust_isomds_manhattan <- km_isomds_manhattan$cluster

# isoMDS minkowski
isomds_minkowski <- read.csv("results/isomds_minkowski_all.csv") 
km_isomds_minkowski <- kmeans(isomds_minkowski[, c(2, 3)], centers = 7, nstart = 20)
clust_isomds_minkowski <- km_isomds_minkowski$cluster

res_clust <- data.frame(cl$ID, cl$region, clust_lle, clust_umap, clust_rospca, 
                        clust_randpro, clust_mds_sammon, clust_isomds_manhattan, clust_isomds_minkowski)
colnames(res_clust)[2] <- "Region"
write_delim(res_clust, "res_clust_df.txt", delim = "\t")

library(mclust)
adjustedRandIndex(res_clust$Region, res_clust$clust_lle)
adjustedRandIndex(res_clust$Region, res_clust$clust_umap)
adjustedRandIndex(res_clust$Region, res_clust$clust_rospca)
adjustedRandIndex(res_clust$Region, res_clust$clust_randpro)
adjustedRandIndex(res_clust$Region, res_clust$clust_mds_sammon)
adjustedRandIndex(res_clust$Region, res_clust$clust_isomds_manhattan)
adjustedRandIndex(res_clust$Region, res_clust$clust_isomds_minkowski)

# 4- random projection components selection ---- 
library(scales)
nsamples <- 1043
get_c <- function(nsamples, eps){
  ncom <- 4 * log(nsamples) / (eps^2/2 - eps^3/3)
  return(ncom)
}
eps <- seq(0.01, 1, 0.01)
ncoms <- sapply(eps, function(x)get_c(nsamples, x))
plt_eps <- data.frame(eps, (ncoms))
colnames(plt_eps) <- c("eps", "ncoms")
ggplot(plt_eps, aes(x = eps, y = ncoms)) + 
  geom_line(lwd = .8, color = "#377eb8") + 
  geom_point(shape = 21, color = "#636363", size = 1, stroke = .8, fill = "#377eb8") + 
  geom_vline(xintercept = 0.1, linetype = "dashed", color = "#e41a1c") + 
  scale_y_continuous(trans = log10_trans(),
                     breaks = trans_breaks("log10", function(x) 10^x),
                     labels = trans_format("log10", math_format(10^.x)),
                     limits = c(10^2, 10^6)) + 
  scale_x_continuous(breaks = seq(0,1,0.1)) + 
  annotation_logticks(sides = "l")  +
  labs(x = "Distortion eps", y = "Dimensions") + 
  theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), 
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=16,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='plain',color='black'),axis.text.y=element_text(size=14,face='bold',color='black'),
        axis.title.x=element_text(size=18,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=18,face='plain',color='black'))
ggsave("figures/scattered_line_randpro.pdf", width = 5, height = 3.5)
ggsave("figures/scattered_line_randpro.png", width = 5, height = 3.5)

# 5- DR methods comparisons ----
methods <- c('PCA', 'MDS', 't-SNE', 'ISOMAP', 'LLE', 'UMAP', 'ROSPCA', 'RandProj', 'Sammon')
ARI <- c(round(c(0.5829841517165872, 0.5847292296539066, 0.6905735949475522, 0.4636345205636106), 8),
         adjustedRandIndex(res_clust$Region, res_clust$clust_lle),
         adjustedRandIndex(res_clust$Region, res_clust$clust_umap),
         adjustedRandIndex(res_clust$Region, res_clust$clust_rospca),
         adjustedRandIndex(res_clust$Region, res_clust$clust_randpro),
         adjustedRandIndex(res_clust$Region, res_clust$clust_mds_sammon))
plt_DR <- data.frame(methods, ARI)

ggplot(plt_DR, aes(x = reorder(methods, ARI, decreasing = F), y = ARI)) + 
  geom_hline(yintercept = seq(0.1, 0.9, 0.1), linetype = "dashed", color = "#dddddd") + 
  geom_segment( aes(x= reorder(methods, ARI, decreasing = F), xend= reorder(methods, ARI, decreasing = F),
                    y=0, yend=ARI), color="#377eb8", lwd = 1.5) +
  geom_point( color="#e41a1c", size=5, alpha= 1) +
  scale_y_continuous(expand = c(0, 0), breaks = seq(0, 1, 0.1), limits = c(0, 1)) + 
  coord_flip() +
  labs(x = " ", y = "Adjusted Rand Index (ARI) score") + 
  theme_classic()+
  theme(panel.background=element_rect(fill='transparent',color='black',size=1),plot.margin=unit(c(0.5,1,0.5,1),'lines'),
        legend.title = element_blank(), 
        legend.key.width=unit(0.6,'cm'),legend.key.height=unit(0.6,'cm'),legend.key.size = unit(5, 'lines'),legend.key = element_rect(size = 0.1, color = NA),
        legend.position='bottom',legend.text=element_text(size=16,face='bold'),legend.margin=margin(t=0.1,r=0.1,b=0,l=0.1,unit='cm'),
        axis.text.x=element_text(size=14,face='plain',color='black'),axis.text.y=element_text(size=14,face='plain',color='black'),
        axis.title.x=element_text(size=18,vjust=0,hjust=0.5,face='plain',color='black'),axis.title.y=element_text(size=18,face='plain',color='black'))
ggsave("figures/lollipop_ARI.pdf", width = 5, height = 5)
ggsave("figures/lollipop_ARI.png", width = 5, height = 5)
