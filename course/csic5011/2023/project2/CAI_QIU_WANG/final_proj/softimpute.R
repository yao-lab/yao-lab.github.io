

sir <- function(rating_mat){
  library("softImpute")
  set.seed(123)
  rk = 9
  reg = 0.01
  res <- softImpute(rating_mat, rank.max=rk, lambda=reg, trace=TRUE, type="svd") 
  return(res)
}