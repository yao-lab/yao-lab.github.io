library(tidyverse)
library(readxl)

data_all_count <- read.table("./data/GSE104276_all_pfc_2394_UMI_count_NOERCC.xls")
data_all_TPM <- read.table("./data/GSE104276_all_pfc_2394_UMI_TPM_NOERCC.xls")

df_all <- data_all_count
df_GW08 <- df_all[, grepl("^GW08", colnames(df_all))]
df_GW09 <- df_all[, grepl("^GW09", colnames(df_all))]
df_GW10 <- df_all[, grepl("^GW10", colnames(df_all))]
df_GW12 <- df_all[, grepl("^GW12", colnames(df_all))]
df_GW13 <- df_all[, grepl("^GW13", colnames(df_all))]
df_GW16 <- df_all[, grepl("^GW16", colnames(df_all))]
df_GW19 <- df_all[, grepl("^GW19", colnames(df_all))]
df_GW23 <- df_all[, grepl("^GW23", colnames(df_all))]
df_GW26 <- df_all[, grepl("^GW26", colnames(df_all))]
write.table(df_GW08, "data/GW08_23cells_count.txt", sep = "\t", 
            row.names = T, col.names = F,  quote = F)
write.table(df_GW09, "data/GW09_88cells_count.txt", sep = "\t", 
            row.names = T, col.names = F,  quote = F)
write.table(df_GW10, "data/GW10_191cells_count.txt", sep = "\t", 
            row.names = T, col.names = F,  quote = F)
write.table(df_GW12, "data/GW12_88cells_count.txt", sep = "\t", 
            row.names = T, col.names = F,  quote = F)
write.table(df_GW13, "data/GW13_24cells_count.txt", sep = "\t", 
            row.names = T, col.names = F,  quote = F)
write.table(df_GW16, "data/GW16_789cells_count.txt", sep = "\t", 
            row.names = T, col.names = F,  quote = F)
write.table(df_GW19, "data/GW19_120cells_count.txt", sep = "\t", 
            row.names = T, col.names = F,  quote = F)
write.table(df_GW23, "data/GW23_324cells_count.txt", sep = "\t", 
            row.names = T, col.names = F,  quote = F)
write.table(df_GW26, "data/GW26_747cells_count.txt", sep = "\t", 
            row.names = T, col.names = F,  quote = F)

