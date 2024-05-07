rm(list = ls())
library(data.table)
library(dplyr)
library(Rdimtools)
library(findPC)
library(tibble)
library(ggplot2)
library(tidydr)
library(withr)
library(uwot)
setwd("/media/london_A/kewei/MATH5473/")

# Load mutation data
mut.df <- fread("./data/ceph_hgdp_minor_code_XNA.betterAnnotated.csv")
mut.df <- mut.df %>% dplyr::select(-c(chr, pos))
mut.df <- column_to_rownames(mut.df, var = "snp")
mut.df <- as.matrix(mut.df)

# Load meta data
meta.info <- read.csv("data/ceph_hgdp_minor_code_XNA.sampleInformation.csv")

#######################
# Perform PCA directly
pca.res.direct <- prcomp(t(mut.df), center = F, scale = F)
saveRDS(pca.res.direct, "pca.res.direct.rds")
optimal.pc <- findPC(pca.res.direct$sdev, number = 50, method = "second derivative", figure = T)
umap.axis <- with_seed(1234, umap(pca.res.direct$x[,1:optimal.pc]))

pca.axis <- as.data.frame(pca.res.direct$x)
pca.axis$region <- meta.info$region

pdf("pca.direct.pdf", width = 4.5, height = 4)
ggplot(pca.axis, aes(PC1, PC2, color = region))+
  geom_point(size = 1.5)+
  theme_dr()+
  theme(panel.grid = element_blank(),
        axis.title = element_text(size = 15),
        legend.title = element_text(size = 13),
        legend.text = element_text(size = 12))
dev.off()