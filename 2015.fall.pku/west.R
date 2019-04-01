load("west.RData")
dim(west)
s0<-colSums(as.matrix(west))
data<-west[,s0>=20]
summary(data)

X<-as.matrix(2*data[,1:10]-1);

obj = ising(X,10,0.1,nt=1000,trate=100)

library('igraph')
g<-graph.adjacency(obj$path[,,850],mode="undirected",weighted=TRUE)
E(g)[E(g)$weight<0]$color<-"red"
E(g)[E(g)$weight>0]$color<-"green"
#plot(g,vertex.shape="rectangle",vertex.size=24,edge.width=2*abs(E(g)$weight))
V(g)$name<-attributes(data)$names
plot(g,vertex.shape="rectangle",vertex.size=35,vertex.label=V(g)$name,edge.width=2*abs(E(g)$weight),vertex.label.family='STKaiti',main="Ising Model (LB): sparsity=0.51")

library('huge')
obj2<- huge(as.matrix(data), method = "glasso")
# Conducting the graphical lasso (glasso)....done.                                          
obj2.select = huge.select(obj2,criterion = "ebic")
# Conducting extended Bayesian information criterion (ebic) selection....done
g2<-graph.adjacency(as.matrix(obj2.select$opt.icov),mode="plus",weighted=TRUE,diag=FALSE)
E(g2)[E(g2)$weight<0]$color<-"red"
E(g2)[E(g2)$weight>0]$color<-"green"
V(g2)$name<-attributes(data)$names
#plot(g2,vertex.shape="rectangle",vertex.size=24,edge.width=2*abs(E(g2)$weight))
plot(g2,vertex.shape="rectangle",vertex.size=35,edge.width=2*abs(E(g2)$weight),vertex.label=V(g2)$name,vertex.label.family='STKaiti',main="Graphical LASSO: sparsity=0.51")
