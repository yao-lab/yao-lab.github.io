# Elo++ by Quanwu Xiao, Microsoft, Beijing
# Reference: 
#	How I won the “Chess Ratings -- Elo vs the Rest of the World” Competition
#	Yannis Syismanis, arXiv:1012.4571v1

# system.time(source("Elopp.R", echo=TRUE))

training <- as.matrix(read.csv("data/training_data.csv"))


agg1 <- aggregate(x=training[,4],by=list(training[,2]),FUN="sum")
agg2 <- aggregate(x=training[,4],by=list(training[,3]),FUN="sum")

maxplayer <- max(agg1[,1],agg2[,1])
ri <- rep(0,maxplayer)

ngame <- nrow(training)
pred <- rep(0,ngame)

w <- rep(0,ngame)
w <- (training[,1]/max(training[,1]))^2

train <- as.matrix(training)
games <- rbind(train[,2:3], train[,3:2])
nn <- split(games[,1], games[,2])
nnum <- aggregate(games[,1],by=list(games[,2]),FUN="length")
nnn <- rep(0, maxplayer)
nnn[nnum[,1]] <- nnum[,2]

wn <- split(c(w,w), games[,2])

lamd=0.77
gama=0.2

P <- 50
for (p in 1:P)
{
	
	ai <- rep(0,maxplayer)
	for (i in 1:maxplayer)
	{
		k <-  nn[[toString(i)]]
		wk <- wn[[toString(i)]]
		ai[i] <- sum(wk*ri[k])/sum(wk)
	}

	yita <- ((1+0.1*P)/(p+0.1*P))^0.602
	for(i in 1:ngame)
	{
		pred[i] <- 1/(1+exp(ri[training[i,3]]-(ri[training[i,2]]+gama)))
		ri[training[i,2]] <- ri[training[i,2]]-yita*(w[i]*(pred[i]-training[i,4])*pred[i]*(1-pred[i])+lamd/nnn[training[i,2]]*(ri[training[i,2]]-ai[training[i,2]]))
		ri[training[i,3]] <- ri[training[i,3]]-yita*(-w[i]*(pred[i]-training[i,4])*pred[i]*(1-pred[i])+lamd/nnn[training[i,3]]*(ri[training[i,3]]-ai[training[i,3]]))
	}
	# print(head(ri))
}


test <- as.matrix(read.csv("data/test_scores.csv"))

prediction <- 1/(1+exp(ri[test[,3]]-(ri[test[,2]]+gama)))

games_new <- rbind(test[,1:2], test[,c(1,3)])
actual <- c(test[,4], 1-test[,4])
prediction <- c(prediction, 1-prediction)

ra <- aggregate(actual, by=list(games_new[,1], games_new[,2]), FUN=sum)$x
rp <- aggregate(prediction, by=list(games_new[,1], games_new[,2]), FUN=sum)$x
print(sqrt(mean((ra-rp)^2)))
