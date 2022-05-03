library(tidyverse)
library(RSQLite)
library(lubridate) 
library(readxl)

characteristics <- read_csv("GKX_20201231.csv")


# characteristics <- characteristics %>%
#   mutate(DATE = ymd(DATE),
#          DATE = floor_date(DATE, "month")) %>%
#   rename("month" = "DATE") %>%
#   rename_with(~paste0("characteristic_", .), -c(permno, month, sic2, RET))

characteristics <- characteristics %>%
  mutate(DATE = ymd(DATE),
         DATE = floor_date(DATE, "month")) 
# %>%
#   rename("month" = "DATE")

characteristics_names = names(characteristics)
other_names = c("permno", "DATE", "RET", "SHROUT", "sic2", "mve0", "prc")
characteristics_names = setdiff(characteristics_names, other_names)


rank_transform <- function(x){
  rx <- rank(x, na.last = TRUE)
  non_nas <- sum(!is.na(x))
  rx[rx>non_nas] <- NA
  2*(rx/non_nas - 0.5)
}

characteristics <- characteristics %>%
  group_by(DATE) %>%
  mutate(across(all_of(characteristics_names), rank_transform))

replace_nas <- function(x){
  med <- median(x, na.rm = TRUE)
  x[is.na(x)] <- med
  x[is.na(x)] <- 0
  return(x)
}

characteristics <- characteristics %>%
  group_by(DATE) %>%
  mutate(across(all_of(characteristics_names), replace_nas))
 
# characteristics <- characteristics %>%
#   group_by(DATE) %>%
#   group_by(permno) %>%


# write.csv(characteristics, "data_clean.csv")

macropredictors_raw <- read_excel("PredictorData2021.xlsx", na="NaN")

macropredictors_raw <- macropredictors_raw %>%
  mutate(yyyymm = ym(yyyymm),
         yyyymm = floor_date(yyyymm, "month")) %>%
  rename("DATE" = "yyyymm")

macropredictors_names <- c("d_p", "e_p", "b_m", "ntis", "tbl", "tms", "dfy", "svar")

macropredictors <- data.frame("DATE"=macropredictors_raw["DATE"])

macropredictors["d_p"]   <- log(macropredictors_raw["D12"]) - log(macropredictors_raw["Index"])
macropredictors["e_p"]   <- log(macropredictors_raw["E12"]) - log(macropredictors_raw["Index"])
macropredictors["b_m"]   <- macropredictors_raw["b/m"]
macropredictors["ntis"]  <- macropredictors_raw["ntis"]
macropredictors["tbl"]   <- macropredictors_raw["tbl"]
macropredictors["tms"]   <- macropredictors_raw["lty"] - macropredictors_raw["tbl"]
macropredictors["dfy"]   <- macropredictors_raw["BAA"] - macropredictors_raw["AAA"]
macropredictors["svar"]  <- macropredictors_raw["svar"]


rank_transform <- function(x){
  rx <- rank(x, na.last = TRUE)
  non_nas <- sum(!is.na(x))
  rx[rx>non_nas] <- NA
  2*(rx/non_nas - 0.5)
}

macropredictors <- macropredictors %>%
  mutate(across(all_of(macropredictors_names), rank_transform))


replace_nas <- function(x){
  med <- median(x, na.rm = TRUE)
  x[is.na(x)] <- med
  x[is.na(x)] <- 0
  return(x)
}

macropredictors <- macropredictors %>%
  group_by(DATE) %>%
  mutate(across(all_of(macropredictors_names), replace_nas))

write.csv(macropredictors, "macropredictors_clean_rank.csv")