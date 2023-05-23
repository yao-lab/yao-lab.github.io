# Load libraries
library(dplyr)
library(corrplot)
library(Seurat)
## read in data and define constants
setwd(r"(D:\OneDrive - HKUST Connect\Courses\CSIC5011\Project\Project2\data)")
seurat.obj = readRDS('part1.rds')
deng_SCE = as.SingleCellExperiment(seurat.obj)
files.path = 'outputs_part2'
cell_metadata = seurat.obj@meta.data
# Read the data files
slingshot <- read.table(file.path(files.path, 'final_slingshot.txt'), header = TRUE, sep = "\t", stringsAsFactors = FALSE)
names(slingshot)[names(slingshot) == "pseudo_time"] <- "slingshot"
tscan <- read.table(file.path(files.path, 'final_tscan.txt'), header = TRUE, sep = "\t", stringsAsFactors = FALSE)
names(tscan)[names(tscan) == "pseudo_time"] <- "tscan"
monocle <- read.table(file.path(files.path, 'final_mnc.txt'), header = F, sep = "\t", stringsAsFactors = FALSE,skip=1)
colnames(monocle) <- c("cell", "monocle")

cell_metadata$week_n <- as.numeric(sub("GW", "", cell_metadata$week))

cell_metadata$Label <- rank(cell_metadata$week_n)
cell_metadata$UMAP1 <- seurat.obj@reductions$umap@cell.embeddings[,'UMAP_1']
cell_metadata$Cell = colnames(seurat.obj)
# Merge trajectory analysis results with cell metadata
cell_metadata <- cell_metadata %>%
  left_join(slingshot, by = c("Cell" = "cell")) %>%
  left_join(tscan, by = c("Cell" = "cell")) %>%
  left_join(monocle, by = c("Cell" = "cell"))

# Calculate correlations between different pseudotime values and PC1
df_pseudotime <- cell_metadata %>%
  select(UMAP1, tscan, slingshot, monocle,Label)


# Replace infinite traj.coord values with NA
df_pseudotime$monocle[is.infinite(df_pseudotime$monocle)] <- NA

# Calculate correlations between different pseudotime values and UMAP1
cor_matrix <- cor(df_pseudotime, use = "na.or.complete")
# plot
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
png("./correlation_matrix_plot.png")
corrplot(cor_matrix, method="color", col=col(200),  
         type="upper", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
         p.mat = NULL, sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE 
)
dev.off()
