library(Seurat)
library(SingleCellExperiment)
library(gam)
## read in data and define constants
setwd(r"(D:\OneDrive - HKUST Connect\Courses\CSIC5011\Project\Project2\data)")
seurat.obj = readRDS('part1.rds')
deng_SCE = as.SingleCellExperiment(seurat.obj)
files.path = 'outputs_part2'
### preprocess
deng_SCE$UMap_1 = reducedDims(deng_SCE)[['UMAP']][,'UMAP_1']
# Only look at the 1,000 most variable genes when identifying temporally expressesd genes.
# Identify the variable genes by ranking all genes by their variance.
Y <- log2(seurat.obj@assays$RNA@counts + 1)
var1K <- names(sort(apply(Y, 1, var), decreasing = TRUE))[1:1000]
Y <- Y[var1K, ]  # only counts for variable genes


######### slingshot #####
sls = read.csv(file.path(files.path, 'final_slingshot.txt'),sep = '\t')
# Fit GAM for each gene using pseudotime as independent variable.
sls.t <- sls$pseudo_time
names(sls.t) = sls$cell
deng_SCE$sls = sls.t[colnames(deng_SCE)]
sls.gam.pval <- apply(Y[,sls$cell], 1, function(z){
  d <- data.frame(z=z, t=sls.t)
  tmp <- gam(z ~ lo(t), data=d)
  p <- summary(tmp)[4][[1]][1,5]
  p
})

# Identify genes with the most significant time-dependent model fit.
sls.topgenes <- names(sort(sls.gam.pval, decreasing = FALSE))[1:length(sls.gam.pval)]  
write.table(sls.topgenes, file.path(files.path, 'sls.genes.txt'), sep = '\n', quote = F, row.names = F, col.names = F)

######### monocle3 #####
mnc = read.csv(file.path(files.path, 'final_mnc.txt'),sep = '\t')
# Fit GAM for each gene using pseudotime as independent variable.
mnc.t <- mnc$Pseudotime 
names(mnc.t) = mnc$sample_name 

deng_SCE$mnc = mnc.t[colnames(deng_SCE)]
mnc.gam.pval <- apply(Y[,names(mnc.t)], 1, function(z){
  d <- data.frame(z=z, t=mnc.t)
  tmp <- gam(z ~ lo(t), data=d)
  p <- summary(tmp)[4][[1]][1,5]
  p
})

# Identify genes with the most significant time-dependent model fit.
mnc.topgenes <- names(sort(mnc.gam.pval, decreasing = FALSE))[1:length(mnc.gam.pval)]  
write.table(mnc.topgenes, file.path(files.path, 'mnc.genes.txt'), sep = '\n', quote = F, row.names = F, col.names = F)

######### tscan #####
tscan = read.csv(file.path(files.path, 'final_tscan.txt'),sep = '\t')
# Fit GAM for each gene using pseudotime as independent variable.
tscan.t <- tscan$pseudo_time
names(tscan.t) = tscan$cell
deng_SCE$tscan = tscan.t[colnames(deng_SCE)]
tscan.gam.pval <- apply(Y[,names(tscan.t)], 1, function(z){
  d <- data.frame(z=z, t=tscan.t)
  tmp <- gam(z ~ lo(t), data=d)
  p <- summary(tmp)[4][[1]][1,5]
  p
})

# Identify genes with the most significant time-dependent model fit.
tscan.topgenes <- names(sort(tscan.gam.pval, decreasing = FALSE))[1:length(tscan.gam.pval)]  
write.table(tscan.topgenes, file.path(files.path, 'tscan.genes.txt'), sep = '\n', quote = F, row.names = F, col.names = F)

### mnc
mnc.1 = plotExpression(deng_SCE, "HMGB2", x = "UMAP_1", 
                       colour_by = "anno", show_violin = T,
                       show_smooth = F)
plotExpression(deng_SCE, "HMGB2", x = "UMAP_1", 
               colour_by = "mnc", show_violin = T,
               show_smooth = F)  + theme(plot.title = element_text(size = 200), yl)

mnc.2 = plotExpression(deng_SCE, "PPP1R17", x = "UMAP_1", 
                       colour_by = "anno", show_violin = T,
                       show_smooth = F)
plotExpression(deng_SCE, "PPP1R17", x = "UMAP_1", 
               colour_by = "mnc", show_violin = T,
               show_smooth = F)
### sls
sls.1 = plotExpression(deng_SCE, "HBB", x = "UMAP_1", 
                       colour_by = "anno", show_violin = T,
                       show_smooth = F)
plotExpression(deng_SCE, "HBB", x = "UMAP_1", 
               colour_by = "sls_hz", show_violin = T,
               show_smooth = F)
sls.2 = plotExpression(deng_SCE, "HBA1", x = "UMAP_1", 
                       colour_by = "anno", show_violin = T,
                       show_smooth = F)
### tscan
tscan.1 = plotExpression(deng_SCE, "GNG5", x = "UMAP_1", 
                         colour_by = "anno", show_violin = T,
                         show_smooth = F)
plotExpression(deng_SCE, "GNG5", x = "UMAP_1", 
               colour_by = "tscan", show_violin = T,
               show_smooth = F)

tscan.2 = plotExpression(deng_SCE, "VIM", x = "UMAP_1", 
                         colour_by = "anno", show_violin = T,
                         show_smooth = F)
plotExpression(deng_SCE, "VIM", x = "UMAP_1", 
               colour_by = "tscan", show_violin = T,
               show_smooth = F)
library(ggpubr)

pre.path = r"(D:\OneDrive - HKUST Connect\Courses\CSIC5011\Project\Project2\presentation)"
# png(file.path(pre.path, "genes_2.png"), width = 600, height = 800, units = "px", res = 180)
ggarrange(mnc.1+rremove("ylab") + rremove("xlab"), mnc.2+rremove("ylab") + rremove("xlab"), 
          sls.1+rremove("ylab") + rremove("xlab"), sls.2+rremove("ylab") + rremove("xlab"),
          tscan.1+rremove("ylab") + rremove("xlab"), tscan.2+rremove("ylab") + rremove("xlab"), 
          labels = c("MNC.1", "MNC.2", "SLS.1", "SLS.2", "TSCAN.1", "TSCAN.2"),
          ncol = 2, nrow = 3,  common.legend = TRUE, legend="bottom")+
  theme(panel.background = element_rect(fill = "white"))
ggsave(file.path(pre.path, "genes_2.png"), width = 1600, height = 800, units = 'px', dpi=100,) 
ggarrange(mnc.1, mnc.2, 
          sls.1, sls.2,
          tscan.1, tscan.2, 
          labels = c("MNC.1", "MNC.2", "SLS.1", "SLS.2", "TSCAN.1", "TSCAN.2"),
          ncol = 2, nrow = 3,  common.legend = TRUE, legend="bottom")+
  theme(panel.background = element_rect(fill = "white"))
ggsave(file.path(pre.path, "genes_2_full.png"), width = 500, height = 800, units = 'px', dpi=100,) 

ggarrange(mnc.1, sls.1, tscan.1,  mnc.2, sls.2,tscan.2, 
          labels = c("MNC.1", "SLS.1", "TSCAN.1", "MNC.2", "SLS.2", "TSCAN.2"),
          ncol = 3, nrow = 2,  common.legend = TRUE, legend="right")+
  theme(panel.background = element_rect(fill = "white"))
ggsave(file.path(pre.path, "genes_2_hor.png"), width = 1200, height = 800, units = 'px', dpi=100,) 


