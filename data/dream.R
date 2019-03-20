load("dream.RData")
dim(dream)

s0<-colSums(dream)
data<-dream[,s0>=40]
attributes(data)$names

 # [1] "贾赦"       "贾政"       "贾珍"       "贾琏"       "贾宝玉"    
 # [6] "贾环"       "贾元春"     "贾迎春"     "贾探春"     "贾惜春"    
# [11] "贾蓉"       "贾兰"       "贾蔷"       "贾芸"       "贾巧姐"    
# [16] "史太君"     "史湘云"     "王夫人"     "王熙凤"     "薛姨妈"    
# [21] "薛蟠"       "薛蝌"       "薛宝钗"     "薛宝琴"     "林黛玉"    
# [26] "邢夫人"     "尤氏"       "李纨"       "秦氏"       "香菱"      
# [31] "妙玉"       "赵姨娘"     "刘姥姥"     "袭人"       "晴雯"      
# [36] "麝月"       "秋纹"       "紫鹃"       "雪雁"       "鸳鸯"      
# [41] "琥珀"       "莺儿"       "平儿"       "小红"       "彩云"      
# [46] "茗烟"       "李贵"       "芳官"       "贾雨村"     "周瑞家的"  
# [51] "秦钟"       "赖大"       "林之孝"     "林之孝家的" "邢岫烟"    
# [56] "尤二姐"    

p = dim(data)[2];
X<-as.matrix(2*as.matrix(data[,1:p])-1);
library("Libra")
obj = ising(X,10,0.1,nt=1000,trate=100)
image(obj$path[,,500])

sparsity=NULL
for (i in 1:1000) {sparsity[i]<-(sum(abs(obj$path[,,i])>1e-10)-p)/(p^2-p) }

library('igraph')
g<-graph.adjacency(obj$path[,,326],mode="undirected",weighted=TRUE)
E(g)[E(g)$weight<0]$color<-"red"
E(g)[E(g)$weight>0]$color<-"green"
#plot(g,vertex.shape="rectangle",vertex.size=24,edge.width=2*abs(E(g)$weight))
V(g)$name<-attributes(data)$names
plot(g,vertex.shape="rectangle",vertex.size=25,vertex.label=V(g)$name,edge.width=2*abs(E(g)$weight),vertex.label.family='STKaiti',main="Ising Model (LB): sparsity=20%")

# Choose the first 80 chapters
data<-dream[dream.data[,1]>0,]
dim(data)
s0<-colSums(data)
data1<-data[,s0>=30]
attributes(data1)$names

p = dim(data1)[2];
X<-as.matrix(2*as.matrix(data1[,1:p])-1);
obj = ising(X,10,0.1,nt=1000,trate=100)
image(obj$path[,,500])

sparsity=NULL
for (i in 1:1000) {sparsity[i]<-(sum(abs(obj$path[,,i])>1e-10)-p)/(p^2-p) }
sparsity

library('igraph')
g<-graph.adjacency(obj$path[,,621],mode="undirected",weighted=TRUE)
E(g)[E(g)$weight<0]$color<-"red"
E(g)[E(g)$weight>0]$color<-"green"
#plot(g,vertex.shape="rectangle",vertex.size=24,edge.width=2*abs(E(g)$weight))
V(g)$name<-attributes(data1)$names
plot(g,vertex.shape="rectangle",vertex.size=25,vertex.label=V(g)$name,edge.width=2*abs(E(g)$weight),vertex.label.family='STKaiti',main="Ising Model (LB): sparsity=50%")

# Choose the later 40 chapters
data<-dream[dream.data[,1]<1,]
dim(data)
data2<-data[,attributes(data1)$names]
attributes(data2)$names

p = dim(data2)[2];
X<-as.matrix(2*as.matrix(data2[,1:p])-1);
obj = ising(X,10,0.1,nt=1000,trate=100)
image(obj$path[,,500])

sparsity=NULL
for (i in 1:1000) {sparsity[i]<-(sum(abs(obj$path[,,i])>1e-10)-p)/(p^2-p) }
sparsity

library('igraph')
g<-graph.adjacency(obj$path[,,598],mode="undirected",weighted=TRUE)
E(g)[E(g)$weight<0]$color<-"red"
E(g)[E(g)$weight>0]$color<-"green"
#plot(g,vertex.shape="rectangle",vertex.size=24,edge.width=2*abs(E(g)$weight))
V(g)$name<-attributes(data2)$names
plot(g,vertex.shape="rectangle",vertex.size=25,vertex.label=V(g)$name,edge.width=2*abs(E(g)$weight),vertex.label.family='STKaiti',main="Ising Model (LB): sparsity=50%")


