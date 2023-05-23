setwd(r"(D:\OneDrive - HKUST Connect\Courses\CSIC5011\Project\Project2\data)")
### read in preprocessed data
seurat_obj = readRDS('part1.rds')
# define output path
output.path = 'outputs_part2'
# get meta data
meta.data = seurat.obj@meta.data
# convert week string to number, e.g., "week08" -> 8
week.num = numeric_value <- as.numeric(gsub("GW", "", meta.data$week))
names(week.num) = rownames(meta.data)
week.info = as.data.frame(week.num)
week.info$sample_name = rownames(week.info)
############# Monocle 3 ############
library("monocle3")
### generate gt_anno.txt
# anno = read_excel(file.path(data.path, 'GSE104276_readme_sample_barcode.xlsx'),sheet = 'SampleInfo')
# colnames(anno)[1] = 'name'
# write.table(anno, file.path(out.path, 'gt_anno.txt'), quote = F, row.names = F, col.names = T, sep = '\t')

# read in ground truth annotation for the cells. 
anno_gt<- read.table("gt_anno.txt",header = T,sep = "\t",row.names = 1)
dim(anno_gt)
anno<- read.table("./meta.data.Serat.Cluster.txt",header = T,sep = "\t",row.names = 1)

CellS<- intersect(rownames(anno_gt),colnames(pbmc.data))

expression_matrix<- seurat.obj@assays$RNA@counts[,CellS]
dim(expression_matrix)
anno<- anno[CellS,]
dim(anno)


gene_annotation<- data.frame(Gene=rownames(expression_matrix))
rownames(gene_annotation)<- rownames(expression_matrix)
gene_annotation$gene_short_name<- gene_annotation$Gene
row.names(anno) = colnames(expression_matrix)
cds <- new_cell_data_set(as.matrix(expression_matrix),
                         cell_metadata = anno,
                         gene_metadata = gene_annotation)

cds <- cds[,Matrix::colSums(exprs(cds)) != 0]
cds <- estimate_size_factors(cds)
cds <- preprocess_cds(cds, num_dim = 50)

# Reduce dimensionality
cds <- reduce_dimension(cds,reduction_method="UMAP")
# Clustering cells
cds <- cluster_cells(cds,reduction_method="UMAP")
# learn_graph
cds <- learn_graph(cds)
# Order the cells in pseudotime and determined the starting point
cds <- order_cells(cds)
saveRDS(cds, 'cds.rds')
# save plots 
plot_cells(cds, label_groups_by_cluster=T,  color_cells_by = "CellType", label_branch_points = F, group_label_size = 3, label_leaves = T, label_roots=F)
ggsave(file.path(pre.path, "celltype.png"), width = 1000, height = 600, units = 'px', dpi=200,) 

plot_cells(cds, label_groups_by_cluster=T,  color_cells_by = "week",label_branch_points = F, group_label_size = 3, label_leaves = F, label_roots=F)
ggsave(file.path(pre.path, "week.png"), width = 1000, height = 600, units = 'px', dpi=200,) 

#### save pseudo time
mnc.df<- cds@principal_graph_aux@listData[["UMAP"]][["pseudotime"]]
mnc.df<- as.data.frame(mnc.df)
rownames(mnc.df) = colnames(cds)
length(interaction(names(week.num), rownames(mnc.df)))

mnc.df = merge(mnc.df, week.info, by = "row.names", all = TRUE)
head(mnc.df)
sum(is.na(mnc.df$sample_name))
sum(is.na(mnc.df$State))
sum(is.na(mnc.df$Pseudotime))
sum(is.na(mnc.df$week.num))
dim(mnc.df)
mnc.df <- na.omit(mnc.df)
dim(mnc.df)
head(mnc.df)
colnames(mnc.df) = c('cell','pseudo_time','gt','sample_name')
write.table(mnc.df, file.path(output.path, 'final_mnc.txt') , quote = F, sep = '\t', row.names = T, col.names = T)


################ Slingshot #################
library(SingleCellExperiment)
library(Seurat)
library(slingshot)
library(Polychrome)
#  conver from seurat
deng_SCE = as.SingleCellExperiment(seurat.obj)
# run slingshot
deng_SCE <- slingshot(deng_SCE, clusterLabels = 'anno',reducedDim = "PCA",
                      allow.breaks = FALSE)
# get summary
summary(deng_SCE$slingPseudotime_1)
lnes <- getLineages(reducedDim(deng_SCE,"PCA"),
                    deng_SCE$anno)
# plot trajectory
my_color <- createPalette(10, c("#010101", "#ff0000"), M=1000)
plot(reducedDims(deng_SCE)$PCA, col = my_color[as.factor(deng_SCE$anno)], 
     pch=16, 
     asp = 1)
legend("bottomleft",legend = names(my_color[levels(deng_SCE$anno)]),  
       fill = my_color[levels(deng_SCE$anno)])
lines(SlingshotDataSet(deng_SCE), lwd=2, type = 'lineages', col = c("black"))

slingshot_df = as.data.frame(deng_SCE[[c('slingPseudotime_1')]])
rownames(slingshot_df) = colnames(deng_SCE)
slingshot_df = na.omit(slingshot_df)
colnames(slingshot_df) = c('pseudo_time')
sls.merged = merge(slingshot_df, week.num, by = "row.names", all = F)
colnames(sls.merged) = c('cell','pseudo_time','gt')
write.table(sls.merged, file.path(output.path, 'final_slingshot.txt') , quote = F, sep = '\t', row.names = T, col.names = T)

################ TSCAN #################
library(ggplot2)
library(TSCAN)
library(ggbeeswarm)
# get counts from preprocessed data
seuratdf<-as.matrix(seurat.obj@assays$RNA@counts)
# preprocess
procdata <- preprocess(seuratdf,cvcutoff = 0)
dim(procdata)
# clustering
lpsmclust <- exprmclust(procdata)
# show clusters
plotmclust(lpsmclust,show_cell_names = F)
# order cells
lpsorder <- TSCANorder(lpsmclust)

### output pseudo time
tscan.df = TSCANorder(lpsmclust,flip=FALSE,orderonly=FALSE)
rownames(tscan.df) = tscan.df$sample_name
length(interaction(names(week.num), tscan.df$sample_name))
week.info$sample_name = rownames(week.info)
tscan.merged = merge(tscan.df, week.info, by = "sample_name", all = TRUE)
head(tscan.merged)
sum(is.na(tscan.merged$sample_name))
sum(is.na(tscan.merged$State))
sum(is.na(tscan.merged$Pseudotime))
sum(is.na(tscan.merged$week.num))
dim(tscan.merged)
tscan.merged <- na.omit(tscan.merged)
dim(tscan.merged)
head(tscan.merged)
colnames(tscan.merged) = c('cell','cluster_num','pseudo_time','gt')
write.table(tscan.merged, file.path(output.path, 'final_tscan.txt') , quote = F, sep = '\t', row.names = T, col.names = T)

