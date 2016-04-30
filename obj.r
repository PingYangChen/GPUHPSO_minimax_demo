

obj <- function(x,y) {
  (x-1)^2 - y^2
}


xg <- seq(-10,10,length=1001)
yg <- seq(-10,10,length=1001)

xtmp <- numeric(length(xg))
jtmp <- numeric(length(xg))
for (i in 1:length(xg)) {
  tmp <- sapply(1:length(yg), function(j) obj(xg[i], yg[j]))
  jtmp[i] <- which.max(tmp)  
  xtmp[i] <- tmp[jtmp[i]]
  
}

xg[which.min(xtmp)]
yg[jtmp[which.min(xtmp)]]

