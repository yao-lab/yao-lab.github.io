
output_Z <- function(mf_result){
  Z <- sapply(mf_result, FUN = function(x){x$mu})
  return(Z)
}


output_W <- function(mf_result){
  W <- sapply(mf_result, FUN = function(x){x$nu})
  return(W)
}