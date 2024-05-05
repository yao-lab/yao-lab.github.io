#import libraries 
library(readxl)
library(dplyr)
library(Seurat)
library(SingleCellExperiment)
library(patchwork)
library(ggplot2)
library(stringr)
library(RColorBrewer)
library(monocle3)
library(monocle)

#### Input the TPM count matrix and start processing scRNA-seq using Seurat package ----
setwd("C:/Users/Jooran/OneDrive - HKUST Connect/CSIC5011")
#TPM values were merged from individual GW dataset
data <- read.csv('TPM_count_matrix.csv') %>% as.data.frame()
rownames(data) <- data$Gene
#get the clinical info 
info <- read.table('metadata.txt', header= T, sep = '\t', row.names = 1) #2309 7
#only get the filtered cells (that has the metadata info)
data <- data[, rownames(info)] #24153  2309


##### plot the barplot (data visualization) ----
info$GW <- substr(rownames(info), 1, 4)
table <- table(info$GW) %>% as.data.frame()
ggplot(table, aes(x=Var1, y = Freq, fill = Var1)) + 
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set1") +
  theme(legend.position="none") + 
  labs(x = "Gestational Weeks", y = "number of cells") + 
  ggtitle('Number of cells of each Gestational Weeks') + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))


##### Perform Seurat Analysis ----
# Initialize the Seurat object with the raw (non-normalized data).
pbmc <- CreateSeuratObject(counts = data, meta.data=info)
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)
pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(pbmc), 10)
# plot variable features with and without labels
plot1 <- VariableFeaturePlot(pbmc)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

# Scaling the data
all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)
# Perform linear dimensional reduction
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
# Examine and visualize PCA results a few different ways
print(pbmc[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")
DimPlot(pbmc, reduction = "pca")
DimHeatmap(pbmc, dims = 1, cells = 500, balanced = TRUE)
DimHeatmap(pbmc, dims = 1:15, cells = 500, balanced = TRUE)

#non-linear dimensional reduction 
#select the appropriate number of dimensions
pbmc <- JackStraw(pbmc, num.replicate = 100)
pbmc <- ScoreJackStraw(pbmc, dims = 1:20)
JackStrawPlot(pbmc, dims = 1:15)
ElbowPlot(pbmc)


###perform hierarchical clustering 
pbmc <- FindNeighbors(pbmc, dims = 1:15)
pbmc <- FindClusters(pbmc, resolution = 0.5)
# Look at cluster IDs of the first 5 cells
head(Idents(pbmc), 5)
pbmc <- RunUMAP(pbmc, dims = 1:10)
pbmc <- RunTSNE(pbmc, dims = 1:10)


#identify the marker genes of each cluster and assign each cluster into one subtype 
Idents(pbmc) <- as.factor(pbmc$seurat_clusters)
pbmc.markers <- FindAllMarkers(pbmc, test.use = "wilcox", only.pos = TRUE,
                               min.pct = 0.25, logfc.threshold = 0.25, 
                               return.thresh = 0.05)
dim(pbmc.markers) # 20326    7
pbmc.markers %>% group_by(cluster) %>% top_n(n = 2, wt = avg_log2FC)

#examine canonical marker gene expressions 
top <- pbmc.markers[rownames(pbmc.markers) %in% c("PTPRC","P2RY12",
                                                  "PAX6","SFRP1",
                                                  "PDGFRA","COL20A1",
                                                  "NEUROD2","RBFOX1",
                                                  "GAD1","PDE4DIP",
                                                  "GFAP","AQP4"), ]

#plot those marker genes only 
DoHeatmap(pbmc, features = top$gene, lab = TRUE)+
  theme(axis.text.y = element_text(size =10))
dev.off()


#add the cell-type label (based on the observations above)
pbmc@meta.data$cell_type2 <- 'NA'
pbmc@meta.data[pbmc@meta.data$seurat_cluster == '0' | 
                 pbmc@meta.data$seurat_cluster == '3' | 
                 pbmc@meta.data$seurat_cluster == '2' | 
                 pbmc@meta.data$seurat_cluster == '6',]$cell_type2 <- 'Excitatory Neurons'
pbmc@meta.data[pbmc@meta.data$seurat_cluster == '5' | 
                 pbmc@meta.data$seurat_cluster == '1' | 
                 pbmc@meta.data$seurat_cluster == '8',]$cell_type2 <- 'InterNeurons'
pbmc@meta.data[pbmc@meta.data$seurat_cluster == '9',] $cell_type2 <- 'OPC'
pbmc@meta.data[pbmc@meta.data$seurat_cluster == '7',] $cell_type2 <- 'Astrocytes'
pbmc@meta.data[pbmc@meta.data$seurat_cluster == '10',] $cell_type2 <- 'microglia'
pbmc@meta.data[pbmc@meta.data$seurat_cluster == '6' | 
                 pbmc@meta.data$seurat_cluster == '4',]$cell_type2 <- 'Stem Cells'

#save the marker gene expression results
#write.table(pbmc.markers,file = "pbmc.markers.seurat.cluster.DE.txt",col.names = T,row.names = T,sep = "\t",quote = F)

Idents(pbmc) <- as.factor(pbmc$cell_type2)
VlnPlot_Key_Gene<- VlnPlot(pbmc,  
                           features = c("PTPRC","P2RY12",
                                        "PAX6","SFRP1",
                                        "PDGFRA","COL20A1",
                                        "NEUROD2","RBFOX1",
                                        "GAD1","PDE4DIP",
                                        "GFAP","AQP4"),
                           pt.size=0.08)
plot(VlnPlot_Key_Gene, ncol = 2)
dev.off()

# Finding differentially expressed features (cluster biomarkers) 
# find markers for every cluster compared to all remaining cells, report only the positive ones
pbmc.markers <- FindAllMarkers(pbmc, test.use = "wilcox", only.pos = TRUE,
                               min.pct = 0.25, logfc.threshold = 0.25, 
                               return.thresh = 0.05)
dim(pbmc.markers) # 10262    7

DimPlot(pbmc, reduction = "tsne",label=T, group.by="cell_type2")


##### Perform the trajectory analysis ---- 
#### 1. Monocle3 ----
#setwd("C:/Users/Jooran/OneDrive - HKUST Connect/CSIC5011")
#prepare the input data for monocle3
info2 <- read.csv('metadata_clustering_seurat.csv')
rownames(info2) <- info2$X
info2$X <- NULL

#retrieve the cells that are overlapped by info, data
cells <- intersect(rownames(info2), colnames(data)) #2309 
dim(data) #24153 2309
dim(info2) #2309 8
#add info (week)
info2$week <- substr(info2$week,3, 4)

expr_matrix <- pbmc@assays$RNA$counts %>% as.data.frame()
gene_annotation <- data.frame(Gene = rownames(expr_matrix))
rownames(gene_annotation) <- rownames(expr_matrix)
gene_annotation$gene_short_name<- gene_annotation$Gene
expr_matrix <- expr_matrix[rownames(expr_matrix) %in% rownames(gene_annotation),] %>%
  as.matrix()
dim(expr_matrix)
dim(gene_annotation)

cds <- new_cell_data_set(expr_matrix, 
                         cell_metadata = info2, 
                         gene_metadata = gene_annotation)
cds <- preprocess_cds(cds, num_dim = 15)



# Reduce dimensionality and visualize the results
cds <- reduce_dimension(cds,reduction_method="UMAP")
# Perform the clustering 
cds <- cluster_cells(cds,reduction_method="UMAP")

#plot learn_graph
cds <- learn_graph(cds)
#based on "cell_type"
UMAP_by_cell_types<- plot_cells(cds, group_cells_by = "cell_type2",
                                color_cells_by = "cell_type2",
                                label_groups_by_cluster=T,
                                label_leaves=F,
                                label_branch_points=FALSE, 
                                cell_size = 1,  group_label_size = 8)

plot(UMAP_by_cell_types)
dev.off()

#based on "week"
UMAP_by_week<- plot_cells(cds,reduction_method="UMAP",
                          group_cells_by = "GW",
                          color_cells_by = "GW",
                          label_groups_by_cluster=T,
                          label_leaves=F,
                          label_branch_points=FALSE, 
                          cell_size = 1,  group_label_size = 8)
plot(UMAP_by_week)


plot_by_cluster<- plot_cells(cds,
                             group_cells_by = "cluster",
                             color_cells_by = "cluster",
                             label_cell_groups=T,
                             label_leaves=T,
                             label_branch_points=T,
                             graph_label_size=1.5, 
                             cell_size = 1,  group_label_size = 8)
plot(plot_by_cluster)
dev.off()

#get the pseudotime of each cell first
#identify the node
plot_cells(cds,
           color_cells_by = "week",
           label_cell_groups=FALSE,
           label_leaves=TRUE,
           label_branch_points=TRUE,
           graph_label_size=5)
# Order the cells in pseudotime and determined the starting point
cds <- order_cells(cds)

#add pseudotime inforation on the cData(cds)
colData(cds)$pseudotime <-  cds@principal_graph_aux@listData[["UMAP"]]$pseudotime
#write.table(colData(cds),'pseudotime_analysis_samples.txt',row.names = T,col.names = T,sep = "\t",quote = F)
plot_by_pseudotime<- plot_cells(cds,
                                group_cells_by = "pseudotime",
                                color_cells_by = "pseudotime",
                                label_cell_groups=T,
                                label_leaves=FALSE,
                                label_branch_points=FALSE,
                                graph_label_size=1.5, 
                                cell_size = 1,  group_label_size = 4)
#pdf("UMAP.color_by_pseudotime.pdf",width = 5,height = 4)
plot(plot_by_pseudotime)
dev.off()


#calculate the correlation between the "week" and "pseudotime" 
monocle3_pseudo <- as.data.frame(colData(cds)$pseudotime)
colnames(monocle3_pseudo) <- c('pseudo')
monocle3_pseudo$week <- as.numeric(colData(cds)$week)
cor.test(monocle3_pseudo$week, monocle3_pseudo$pseudo, method = 'spearman')
#r = 0.803, p < 2.2e-16
#draw barplot
ggplot(monocle3_pseudo, aes(x = week, y = pseudo, fill = factor(week))) +
  geom_boxplot(aes(group = week)) + 
  xlab("Week") +
  ylab("Pseudotime") +
  scale_fill_discrete(name = "Week") + 
  ggtitle('Pseudotime of each week calculated from the Monocle3') + 
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))


#### 1-1.Monocle3: downstream analysis ---- 
#### find the genes that are differentially expressed on the different paths through the trajectory
test <- graph_test(cds, neighbor_graph="principal_graph", cores=4)
test_suc <- row.names(subset(test, q_value < 0.01))
#order in p-value ascending order
i <- test %>% filter(p_value == 0 ) %>% filter(q_value ==0) %>% 
  filter(status == 'OK') %>% filter(morans_test_statistic > 60) 
i[order(i$morans_I),] %>% head() #135
#NEUROD2
#PAX6
#OLIG1

#those are the gene s that score as highly significant according to graph_test()
plot_cells(cds, genes=c("NEUROD6", "PAX6", "OLIG1", "STMN2", "TOP2A", "RTN1"),
           show_trajectory_graph=F,
           label_cell_groups=F, min_expr = 0.4, 
           label_leaves=F,  cell_size = 0.5) +
  ggtitle('Expression of the known marker with pseudo-time') + 
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))

plot_cells(cds, genes=c("NEUROD6"),
           show_trajectory_graph=F,
           label_cell_groups=F, min_expr = 0.4, 
           label_leaves=F,  cell_size = 1) +
  ggtitle('Expression of the NEUROD6 with pseudo-time') + 
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))


# Convert relevant variables to a data frame
data <- data.frame(pseudotime = colData(cds)$pseudotime, 
                   PAX6 = pbmc@assays$RNA$counts['PAX6',],
                   STMN2 = pbmc@assays$RNA$counts['STMN2',],
                   OLIG1 = pbmc@assays$RNA$counts['OLIG1',],
                   NEUROD6 = pbmc@assays$RNA$counts['NEUROD6',],
                   cell_type = colData(cds)$cell_type2, 
                   week = colData(cds)$week)

library(MASS) 
# Define the custom colors for each week
week_colors <- c("08" = "darkred", 
                 "09" = "darkblue",
                 "10" = "darkgreen", 
                 "12" = "purple", 
                 "13" = "darkorange", 
                 "16" = "skyblue", 
                 "19" = "pink",
                 "23"= 'yellow',
                 "26"= "black")

cell_colors <- c('Astrocytes' = "darkred",
                 'Excitatory Neurons'= "darkblue", 
                 'InterNeurons'= "darkgreen",
                 'Microglia'= "purple", 
                 'OPC'= "darkorange", 
                 'Stem Cells'= "pink")

p <- ggplot(data, aes(x = pseudotime, y = OLIG1, color = as.factor(week))) +
  geom_point() +
  geom_smooth(aes(color = as.factor(week)), method = "gam", color = "black") +
  xlab("Pseudotime") +
  ylab("Expression of OLIG1") +
  xlim(0, 30) +
  scale_color_manual(values = week_colors, name = "Week") +
  ggtitle('Expression of OLIG1 with Pseudotime') +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
# Display the plot with legend
p + guides(color = guide_legend(title = "Week"))


#based on "cell-type"
p <- ggplot(data, aes(x = pseudotime, y = OLIG1, color = as.factor(cell_type))) +
  geom_point() +
  geom_smooth(aes(color = as.factor(week)), method = "gam", color = "black") +
  xlab("Pseudotime") +
  ylab("Expression of OLIG1") +
  xlim(0, 30) +
  scale_color_manual(values = cell_colors, name = "cell_type") +
  ggtitle('Expression of OLIG1 with Pseudotime') +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
# Display the plot with legend
p + guides(color = guide_legend(title = "cell_type"))

p <- ggplot(data, aes(x = pseudotime, y = NEUROD6, color = as.factor(week))) +
  geom_point() +
  geom_smooth(aes(color = as.factor(week)), method = "gam", color = "black") +
  xlab("Pseudotime") +
  ylab("Expression of NEUROD6") +
  xlim(0, 30) +
  scale_color_manual(values = week_colors, name = "Week") +
  ggtitle('Expression of NEUROD6 with Pseudotime') +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
# Display the plot with legend
p + guides(color = guide_legend(title = "Week"))

p <- ggplot(data, aes(x = pseudotime, y = STMN2, color = as.factor(week))) +
  geom_point() +
  geom_smooth(aes(color = as.factor(week)), method = "gam", color = "black") +
  xlab("Pseudotime") +
  ylab("Expression of STMN2") +
  xlim(0, 30) +
  scale_color_manual(values = week_colors, name = "Week") +
  ggtitle('Expression of STMN2 with Pseudotime') +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
# Display the plot with legend
p + guides(color = guide_legend(title = "Week"))


#### 2. Diffusion map #### ----
library(SingleCellExperiment)
library(destiny)
library(ggthemes)
library(ggbeeswarm)
#convert seurat > sce component
#pbmc <- readRDS('pbmc.rds')
pbmc_sce <- as.SingleCellExperiment(pbmc)
deng <- logcounts(pbmc_sce)
cellLabels <- pbmc@meta.data$cell_type2
colnames(deng) <- cellLabels

# Make a diffusion map.
deng <- as.matrix(deng)
dm <- DiffusionMap(t(deng))
head(colData(pbmc_sce))
# Plot diffusion component 1 vs diffusion component 2 (DC1 vs DC2). 
tmp <- data.frame(DC1 = eigenvectors(dm)[, 1],
                  DC2 = eigenvectors(dm)[, 2],
                  Timepoint = cellLabels)
ggplot(tmp, aes(x = DC1, y = DC2, colour = Timepoint)) +
  geom_point() + scale_color_tableau() + 
  xlab("Diffusion component 1") + 
  ylab("Diffusion component 2") +
  theme_classic() 

# Next, let us use the first diffusion component (DC1) as a measure of pseudotime.
# How does the separation by cell stage look?
pbmc_sce$pseudotime_diffusionmap <- rank(eigenvectors(dm)[,1])    # rank cells by their dpt
ggplot(as.data.frame(colData(pbmc_sce)), 
       aes(x = pseudotime_diffusionmap, 
           y = cell_type2, colour = cell_type2)) +
  geom_quasirandom(groupOnX = FALSE) +
  scale_color_tableau() + theme_classic() +
  xlab("Diffusion component 1 (DC1)") + ylab("Timepoint") +
  ggtitle("Cells ordered by DC1")


# What happens if you run the diffusion map on the PCs? Why would one do this?
# Run PCA on Deng data. Use the runPCA function from the SingleCellExperiment package.
library(scater)
pbmc_sce <- runPCA(pbmc_sce, ncomponents = 50)

# Use the reducedDim function to access the PCA and store the results. 
pca <- reducedDim(pbmc_sce, "PCA")
rownames(pca) <- cellLabels
dm <- DiffusionMap(pca)
# Diffusion pseudotime calculation. 
# Set index or tip of pseudotime calculation to be a zygotic cell (cell 268). 
dpt <- DPT(dm, tips = 100)

# Plot DC1 vs DC2 and color the cells by their inferred diffusion pseudotime.
# We can accesss diffusion pseudotime via dpt$dpt.
df <- data.frame(DC1 = eigenvectors(dm)[, 1], DC2 = eigenvectors(dm)[, 2], 
                 dptval = dpt$dpt, #eigenvalue
                 cell_type2 = cellLabels, week = pbmc@meta.data$week)

df$week2 <- as.numeric(substr(df$week, 3, nchar(df$week)))
p1 <- ggplot(df) + geom_point(aes(x = DC1, y = DC2, color = dptval))
p2 <- ggplot(df) + geom_point(aes(x = DC1, y = DC2, color = cell_type2))
library(cowplot)
p <- plot_grid(p1, p2)
p+ ggtitle('Pseudotime of each week calculated from the Diffusion map') + 
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
#calculate the correlation between the "week" and "pseudotime" 
dmap_pseudo <- df
cor.test(dmap_pseudo$week2, dmap_pseudo$dptval, method = 'spearman')
#r = 0.447, p < 2.2e-16
#draw barplot
ggplot(dmap_pseudo, aes(x = week2, y = dptval, fill = factor(week2))) +
  geom_boxplot(aes(group = week2)) + 
  xlab("Week") +
  ylab("Pseudotime") +
  scale_fill_discrete(name = "Week") + 
  ggtitle('Pseudotime of each week calculated from the Diffusion map') + 
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))





#### 3. Slingshot ####-----
#install and import the slingshot library 
#BiocManager::install('slingshot')
library('slingshot')
sce <- slingshot(pbmc_sce, reducedDim = 'PCA')  # no clusters
# Plot PC1 vs PC2 colored by Slingshot pseudotime.
colors <- rainbow(50, alpha = 1)
plot(reducedDims(sce)$PCA, 
     col = colors[cut(sce$slingPseudotime_1,breaks=50)], pch=16, asp = 1)
lines(SlingshotDataSet(sce), lwd=2)

# Plot Slingshot pseudotime vs cell stage. 
ggplot(as.data.frame(colData(pbmc_sce)), 
       aes(x = sce$slingPseudotime_1, y = cell_type2, 
           colour = cell_type2)) +
  geom_quasirandom(groupOnX = FALSE) +
  scale_color_tableau() + theme_classic() +
  xlab("Slingshot pseudotime") + ylab("Timepoint") +
  ggtitle("Cells ordered by Slingshot pseudotime")

#calculate the correlation between the "week" and "pseudotime" 
sling_pseudo <- data.frame(psuedo = sce$slingPseudotime_1, 
                           cell_type2 = cellLabels,
                           week = pbmc@meta.data$week)
sling_pseudo$week2 <- as.numeric(substr(sling_pseudo$week, 3, nchar(df$week)))
cor.test(sling_pseudo$week2, sling_pseudo$psuedo, method = 'spearman')
#r = 0.5432, p < 2.2e-16
#draw barplot
ggplot(sling_pseudo, aes(x = week2, y = psuedo, fill = factor(week2))) +
  geom_boxplot(aes(group = week2)) + 
  xlab("Week") +
  ylab("Pseudotime") +
  scale_fill_discrete(name = "Week") + 
  ggtitle('Pseudotime of each week calculated from the Slingshot') + 
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))
