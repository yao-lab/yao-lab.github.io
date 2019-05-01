# HodgeRank for chess player rating

# system.time(source("chessHodge1.r", echo=TRUE))
# scoring by hodge

rm(list=ls())
gc()
library(Matrix)

train <- as.matrix(read.csv("../data/training_data.csv"))
valid <- as.matrix(read.csv("../data/cross_validation_dataset.csv"))
test <- as.matrix(read.csv("../data/test_scores.csv"))

players.old <-  sort(unique(c(train[,2], train[,3])))
nplayer <- length(players.old)
max.players.old <- players.old[nplayer]
player.tranform <- rep(0, max.players.old)
player.tranform[1:max.players.old %in% players.old] <- 1:nplayer
train[,2] <- player.tranform[train[,2]]
train[,3] <- player.tranform[train[,3]]
valid[,2] <- player.tranform[valid[,2]]
valid[,3] <- player.tranform[valid[,3]]
test[,2] <- player.tranform[test[,2]]
test[,3] <- player.tranform[test[,3]]


com <- rbind(train[,2], train[,3])
Y <- 2*train[,4]-1	# rescale Y from {1,0.5,0} to {1,0,-1}
w <- train[,1]/max(train[,1])
wY <- Y*w


nMatch <- nrow(train)
comPlus1 <- rbind(com, nplayer+1)
indRow <- rep(1:nMatch, each=3)
indCol <- as.vector(comPlus1)
values <- rep(c(1, -1, 1), nMatch) * rep(w, each=3)
Amat <- sparseMatrix(i=indRow, j=indCol, x=values)


comDouble <- t(cbind(com, com[c(2,1),]))
neighborList <- split(x=comDouble[,1], f=comDouble[,2])
weightList <- split(x=rep(w,2), f=comDouble[,2])
sumNeighborWeight <- lapply(weightList, FUN=sum)
getLijInd <- function(i) {
	v1 <- weightList[[i]]/sumNeighborWeight[[i]]
	v2 <- sapply(split(x=v1, f=neighborList[[i]]), FUN=sum)
	jInd <- as.integer(names(v2))
	return(jInd)
}
getLijValue <- function(i) {
	v1 <- weightList[[i]]/sumNeighborWeight[[i]]
	v2 <- sapply(split(x=v1, f=neighborList[[i]]), FUN=sum)
	names(v2) <- NULL
	LValue <- -v2
	return(LValue)
}
indColList <- lapply(1:nplayer, FUN=getLijInd)
indCol <- unlist(indColList)
indRow <- rep(as.integer(names(neighborList)), times=sapply(indColList, FUN=length))
values <- unlist(lapply(1:nplayer, FUN=getLijValue))
indRow <- c(indRow, 1:nplayer)
indCol <- c(indCol, 1:nplayer)
values <- c(values, rep(1,nplayer))
Lmat <- sparseMatrix(i=indRow, j=indCol, x=values, dims=c(nplayer+1, nplayer+1))

# The following function is used to find optimal regularization parameter
#getValidError <- function(regParam) {
regParam <- 0
valid <- test
Amat <- rBind(Amat, 1)
wY <- c(wY, 0) #avoid ill matrix 
r <- solve(as.matrix(crossprod(Amat)) + regParam*crossprod(Lmat), as.vector(crossprod(Amat, wY)))

predY <- r[valid[,2]] - r[valid[,3]] + r[7302]
predY <- ifelse(predY>1, 1, predY)
predY <- ifelse(predY<-1, -1, predY)
predY <- (predY+1)/2
validY <- valid[,4]
predScores <- rbind(cbind(valid[,c(1,2)], predY), cbind(valid[,c(1,3)], 1-predY))
aggpredScore <- aggregate(predScores[,3], by=list(predScores[,1], predScores[,2]), FUN=sum)
validScores <- rbind(cbind(valid[,c(1,2)], validY), cbind(valid[,c(1,3)], 1-validY))
aggvalidScore <- aggregate(validScores[,3], by=list(validScores[,1], validScores[,2]), FUN=sum)
err <- sqrt(crossprod(aggpredScore[,3] - aggvalidScore[,3])/nrow(aggpredScore))
#return(err)
#}
#sapply(exp(-10:-4), FUN=getValidError)