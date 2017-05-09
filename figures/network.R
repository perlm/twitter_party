setwd('/home/jason/twitter_party/data')
library(reshape2)

library(GGally)
library(network)
library(sna)
library(ggplot2)

# row.names = 1, 
df = read.table("dataframe_political_train.csv", header=TRUE, sep=",", quote='', nrows=100)

df.l <- df[,1:100]

# need to rearrange the file.
df.m <- melt(df.l,id.vars="X")
df.m2 <- subset(df.m,value>0)
df.m3 <- df.m2[,c("X","variable")]
df.m4 <- merge(df.m3,df.m3,by="X")
df.m5 <- subset(df.m4,variable.x!=variable.y)[,2:3]
rm(df.m,df.m2,df.m3,df.m4)

# following - https://briatte.github.io/ggnet/
net = network(df.m5, directed = FALSE)

# network plot
ggnet2(net, alpha = 0.5, size = 3, edge.alpha = 0.25,
       #label=TRUE, label.size = 3,label.color='blue'
       label = c("Eugene_Robinson","POTUS","TheOnion","SarahPalinUSA","hardball","AriMelber","WhoopiGoldberg","GlennThrush","TheDemCoalition"), 
       label.size = 5,label.color='blue') + 
  ggtitle('Clustering of Political Twitter Accounts by Followers')



