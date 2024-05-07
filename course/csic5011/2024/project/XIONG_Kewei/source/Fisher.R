#' @description
#' Identify differential SNPs and colocalization
#' 
rm(list = ls())
library(dplyr)
library(xQTLbiolinks)
library(progress)
setwd("/media/london_A/kewei/MATH5473")

num_cores <- 40
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Load cluster information
cluster <- readRDS('cluster.rds')

# Load mutation data
mut.df <- fread("./data/ceph_hgdp_minor_code_XNA.betterAnnotated.csv")
# mut.df <- mut.df %>% dplyr::select(-c(chr, pos))
mut.df <- column_to_rownames(mut.df, var = "snp")
mut.df <- as.matrix(mut.df)

mut.df1 <- as.data.frame(mut.df)
mut.df1 <- mut.df1 %>% dplyr::select(-c(chr, pos))
mut.df1 <- as.data.frame(t(mut.df1))
mut.df1$Cluster <- cluster$Cluster

# Summarize mutation frequency for each cluster
mut.df1.sub <- mut.df1[,c(sample(1:ncol(mut.df1), 100000, replace = F), ncol(mut.df1))]  # Subsample SNPs to speed up analysis
df_long <- withr::with_seed(1234, mut.df1.sub) %>% 
  pivot_longer(cols = -Cluster, names_to = "mutation", values_to = "status")
mut.freq <- df_long %>% 
  group_by(Cluster, mutation) %>% 
  dplyr::summarise(Mutant = sum(status == 1 | status == 2), Wild = sum(status == 0))

test.res <- data.frame()
for(i in 1:ncol(mut.df1)){
  snp.tmp <- colnames(mut.df1)[i]
  mut.freq.tmp <- mut.freq[mut.freq$mutation == snp.tmp,]
  test.res.tmp <- fisher.test(mut.freq.tmp[,c(3,4)])
  test.res <- rbind(test.res, data.frame(SNP = snp.tmp, Pval = test.res.tmp$p.value))
}



test.res <- data.frame()
pb <- progress_bar$new(total = ncol(mut.df1.sub))
for(i in 1:ncol(mut.df1.sub)){
  snp.tmp <- colnames(mut.df1.sub)[i]
  mut.freq.tmp <- mut.freq[mut.freq$mutation == snp.tmp,]
  test.res.tmp <- fisher.test(mut.freq.tmp[,c(3,4)])
  test.res <- rbind(test.res, data.frame(SNP = snp.tmp, Pval = test.res.tmp$p.value))
  pb$tick()
}
pb$close()
# Correct p-value
test.res$Padj <- p.adjust(test.res$Pval, method = "fdr")
saveRDS(test.res, "diff.snp.res.rds")

sig.freq <- data.frame(Type = c("Diff", "None"),
                       Num = c(nrow(test.res[test.res$Padj < 0.05,]),
                               nrow(test.res[test.res$Padj >= 0.05,])))
sig.freq$Num <- log10(sig.freq$Num)

p <- ggplot(sig.freq, aes(Type, Num))+
  geom_col(width = 0.6)+
  theme_classic2()+
  theme(axis.title = element_text(size = 14, color = "black"),
        axis.text = element_text(size = 13, color = "black"))+
  labs(x = "", y = "log10(Number)")
ggsave("diff.snp.num.pdf", p, width = 3.5, height = 3)

#########################################################
## Colocalization
gwas <- fread("/media/bora_A/zhangt/GWAS_resource/coloc_hg38/Cancer/BC-EUR-Jiang2021NG/GWAS_BC-EUR-Jiang2021NG_COLOC.txt")
gene <- "18807424-18812468" # KLHDC7A
pos.range <- range(18807424 - 1000000, 805231 + 18812468)
gwas.gene <- gwas %>% dplyr::filter(chr == 1 & position >= pos.range[1] & position <= pos.range[2])

# Annotation
share.snp <- intersect(gwas.gene$rsid, test.res$SNP)
gwas.gene <- gwas.gene[match(share.snp, gwas.gene$rsid),]
colnames(gwas.gene) <- c("chrom", "position", "chr_position", "rsid", 'beta', "se", "N", "pValue", "maf")
gwas.gene$chrom <- paste0("chr", gwas.gene$chrom)
gwas.gene <- gwas.gene[,c("rsid", "chrom", "position", "pValue", "maf", "beta", "se")]

#get genomic position
snp.df <- test.res[match(share.snp, test.res$SNP),]
snp.df$chrom <- gwas.gene$chrom
snp.df$position <- gwas.gene$position
snp.df$maf = 0.5; snp.df$beta = 0.5; snp.df$se = 0.5
colnames(snp.df) <- c("rsid", "pValue", "Padj", "chrom", "position", "maf", "beta", "se")
snp.df <- snp.df[,c("rsid", "chrom", "position", "pValue", "maf", "beta", "se")]
# snp.df$chrom <- paste0("chr", snp.df$chrom)
# 
# coloc.res <- xQTLanalyze_coloc_diy(gwasDF = as.data.table(gwas.gene),
#                                    qtlDF = as.data.table(snp.df),
#                                    mafThreshold = 0.01,
#                                    gwasSampleNum = 50000,
#                                    qtlSampleNum = 10000,
#                                    method = "coloc",
#                                    bb.alg = FALSE
#                                    )
# saveRDS(gwas.gene, "gwas.gene.rds")
# saveRDS(snp.df, "snp.df.rds")



# Example
snp.example <- "rs2250072"
x <- as.data.frame(mut.df[snp.example,])
x <- x[-c(1,2),,drop = F]
colnames(x) <- snp.example
x$rs2250072 <- ifelse(x == 0, "Wild", "Mutant")

rp.res <- readRDS("RandomProjection.res")
pca.rp.res <- prcomp(rp.res$Y, center = T, scale = T)
pca.axis <- as.data.frame(pca.rp.res$x)
pca.axis$Cluster <- cluster$Cluster
pca.axis$mutation <- x[,1]
pdf("pca.rs2250072.pdf", width = 4.5, height = 4)
ggplot(pca.axis, aes(PC1, PC2, color = mutation))+
  scale_color_manual(values = c("Mutant" = "darkred", "Wild" = "grey"))+
  geom_point(size = 1.5)+
  theme_dr()+
  theme(panel.grid = element_blank(),
        axis.title = element_text(size = 15),
        legend.title = element_text(size = 13),
        legend.text = element_text(size = 12))
dev.off()