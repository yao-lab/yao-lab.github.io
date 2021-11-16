rm(list = ls()) 

library(data.table) 
library(dplyr) 
library(reshape2) 
library(fastDummies)
#Read data   

###If you have the dataset locally: 
####Method 1: read data from local. Set the path to the location of your data 

path ='D:/hkust_study/6010z/project2/data_project2'

pathdata = paste0(path,'/GKX_20201231.csv')
dta_sub_ch = fread(pathdata)
memory.limit(15000)
dta_sub_ch[,'DATE'] = as.Date(as.character(dta_sub_ch$DATE),'%Y%m%d')


#####choose data between the march 1957 to december 2016 
start_date = as.Date('19570301','%Y%m%d')
end_date = as.Date('20170101','%Y%m%d')
dta_sub_ch = subset(dta_sub_ch, DATE >start_date & DATE <end_date)
dta_sub_ch[,'DATE_ym'] = format(as.Date(dta_sub_ch$DATE),'%Y-%m')


###############missing data for characteristics data  ***
##First step: delete dates that has variables with all na
all_na_ind <- function(x){
  n_x = length(x) 
  n_na = sum(is.na(x))   
  #ind = 0 means vector x is not all na value
  ind = 0
  if(n_x == n_na){
    ind = 1
  }  
  return(ind)
}  

na_ind_dtach = dta_sub_ch %>% group_by(DATE) %>% summarise_all(all_na_ind)



na_ind_dtach = as.data.frame(na_ind_dtach) 


######Replace dates with all nas with 0  

all_na = as.data.frame(colSums(na_ind_dtach[,-1]))

#variables with all na 
var_all_na = colnames(na_ind_dtach)[which(all_na$`colSums(na_ind_dtach[, -1])` != 0)+1 ]

for (var_na in var_all_na){ 
  id_na = which(colnames(dta_sub_ch) == var_na)
  date_na = na_ind_dtach[na_ind_dtach[[id_na]]==1,'DATE'] 
  dta_sub_ch[which(dta_sub_ch$DATE %in% date_na),][[id_na]] = 0
}




######Replace other Nas nas with the median value  

impute_median <- function(x){
  ind_na = is.na(x) 
  x[ind_na] = median(x[!ind_na]) 
  return(as.numeric(x))
}

dta_sub_ch_2 = dta_sub_ch[,-102] %>% group_by(DATE)%>% mutate_all(impute_median)

dta_sub_ch_2[,'DATE_ym']= dta_sub_ch[,'DATE_ym']
rm(dta_sub_ch)

###Choose subset of data 
index_all_sub = unique(dta_sub_ch_2$permno)  


time_all_sub = unique(dta_sub_ch_2$DATE) 

N_sub = length(index_all_sub)
T_sub = length(unique(dta_sub_ch_2$DATE))

N_sub_1 = 500

index_sub_1 =index_all_sub[sample(1:N_sub,N_sub_1)]

dta_sub_ch_3 = subset(dta_sub_ch_2,permno %in% index_sub_1)


Y_all = dta_sub_ch_3[,c('permno','RET','DATE_ym','DATE')]

char_name = colnames(dta_sub_ch_2)[-c(1,2,4,5,6,17,99,102)]

X_all_name = c('permno','DATE_ym',char_name)

X_all = dta_sub_ch_3[,X_all_name]

###
industry = dta_sub_ch_3[,c('permno','sic2')]
industry$sic2 = as.factor(as.integer(industry$sic2))
industry = dummy_cols(industry)




#####Write data as csv
path_data = paste0(path,'/Data_Cleaned')
dir.create(path_data)


####Writing Y  
setwd(path_data) 
#Y 
fwrite(Y_all[,'RET'],'Y_cleaned.csv') 
#X
X_cleaned = cbind(X_all[,-c(1:3)],industry[,-c(1:3)])

fwrite(X_cleaned,'X_cleaned.csv')
#Index date 

fwrite(Y_all[,'DATE'],'Date_list.csv')

fwrite(Y_all[,'permno'],'permno_list.csv')
