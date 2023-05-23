install.packages('Seurat')

library('Seurat')
library('dplyr')
library('patchwork')

###Load data
setwd('E:/MRC/csic/final_project')
data <- read.table(gzfile('./GSE104276_all_pfc_2394_UMI_TPM_NOERCC.xls.gz'))

###Load metadata
meta.data<- read.table("./cell_mapping.txt",header = T,sep = "\t",row.names = 1)

###Create Seurat object
data <- CreateSeuratObject(counts = data, project = "final", min.cells = 3, min.features = 200,meta.data=meta.data)
data <- NormalizeData(data, normalization.method = "LogNormalize", scale.factor = 10000)
data <- NormalizeData(data)

###Identification of highly variable features (feature selection)
data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = 2000)
#Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(data), 10)
#plot variable features with and without labels
plot1 <- VariableFeaturePlot(data)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

###Scaling the data
all.genes <- rownames(data)
data <- ScaleData(data, features = all.genes)

###Perform linear dimensional reduction (PCA)
data <- RunPCA(data, features = VariableFeatures(object = data))
# Examine and visualize PCA results in a few different ways
print(data[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(data, dims = 1:2, reduction = "pca")
DimPlot(data, reduction = "pca")
DimHeatmap(data, dims = 1:15, cells = 500, balanced = TRUE)


###Determine the ‘dimensionality’ of the dataset
data <- JackStraw(data, num.replicate = 100)
data <- ScoreJackStraw(data, dims = 1:20)
JackStrawPlot(data, dims = 1:15)

###Cluster the cells
data <- FindNeighbors(data, dims = 1:15) ####1:x, x is the dimensionality determined in the last step
data <- FindClusters(data, resolution = 0.5)

###Look at cluster IDs of the first 5 cells
head(Idents(data), 5)
data <- RunUMAP(data, dims = 1:10)
data <- RunTSNE(data, dims = 1:10)

DimPlot.do_pre<- DimPlot(data, reduction = "umap",label=T)+ NoLegend()
pdf("Figure.UMAP.Subtype.Without.Annotation.pdf",width = 5,height = 4)
plot(DimPlot.do_pre)
dev.off()
DimPlot(data, reduction = "tsne",label=T,shape.by="cell_types")

###Finding differentially expressed cluster biomarkers
#Find markers for every cluster compared to all remaining cells, report only the positive ones
data.markers <- FindAllMarkers(data, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
dim(data.markers) ### 7461 ### 7 
data.markers %>% group_by(cluster) %>% top_n(n = 2, wt = avg_log2FC)
dim(data.markers)
write.table(data.markers,file = "data.markers.DE.txt",col.names = T,row.names = T,sep = "\t",quote = F)
#Expression heatmap for single cells
top10 <- data.markers %>% group_by(cluster) %>% top_n(n = 50, wt = avg_log2FC)
pdf("Figure.X.DoHeatmap.pdf",width = 14,height = 10)
DoHeatmap(data, features = top10$gene) + NoLegend()
dev.off()
#Visualizing marker expression
pdf("Marker.gene.expression.pdf",width = 14,height = 10)
FeaturePlot(data, features = c("PTPRC","P2RY12","PAX6","SFRP1","OLIG1","PDGFRA","COL20A1","PMP2","NEUROD2","RBFOX1","GAD1","PDE4DIP","GFAP","AQP4","SLCO1C1"))
dev.off()

###Cell type annotation
list_Anno<- list(C0="Excitatory Neurons",C1="Internuerons",C2="NPCs",C3="Excitatory Neurons",C4="Excitatory Neurons",C5="Astrocytes",C6="Excitatory Neurons",C7="OPCs",C8="Microglia")

meta.data.Serat<- data@meta.data
meta.data.Serat$Cell<- rownames(meta.data.Serat)
meta.data.Serat$seurat_clusters<- paste("C",meta.data.Serat$seurat_clusters,sep = "")

Cluster.Anno<- as.data.frame(t(as.data.frame(list_Anno)))
Cluster.Anno$seurat_clusters<- rownames(Cluster.Anno)
colnames(Cluster.Anno)[1]<- "CellType"

Clusmeta.data.Serat.out<- merge(meta.data.Serat,Cluster.Anno,by="seurat_clusters",all=T)
dim(Clusmeta.data.Serat.out)
rownames(Clusmeta.data.Serat.out)<- Clusmeta.data.Serat.out$Cell
write.table(Clusmeta.data.Serat.out,file = "meta.data.Serat.Cluster.txt",col.names = T,row.names = T,sep = "\t",quote = F)

data2<- data
new.cluster.ids <- as.character(list_Anno)
names(new.cluster.ids) <- levels(data2)
data2 <- RenameIdents(data2, new.cluster.ids)

DimPlot.do<- DimPlot(data2, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()

pdf("Figure.UMAP.Subtype.With.Annotation.pdf",width = 5,height = 4)
plot(DimPlot.do) + NoLegend()
dev.off()
