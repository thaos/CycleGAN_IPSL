library(reticulate)
library(magrittr)
library(fields)
library(CDFt)
require (maps)
library(RColorBrewer)

oldpar = par()
resfactor = 2
world = map("world", plot = FALSE)

datasetA  <- py_load_object("datasetA.npy") %>% drop	
datasetB  <- py_load_object("datasetB.npy") %>% drop	
valsetA  <- py_load_object("valsetA.npy") %>% drop	
valsetB  <- py_load_object("valsetB.npy") %>% drop	
histsetA  <- py_load_object("histsetA.npy") %>% drop	
histsetB  <- py_load_object("histsetB.npy") %>% drop	
rcpsetA  <- py_load_object("rcpsetA.npy") %>% drop	
rcpsetB  <- py_load_object("rcpsetB.npy") %>% drop	
fakesetB  <- py_load_object("fakesetB.npy") %>% drop	
fakevalB  <- py_load_object("fakevalB.npy") %>% drop	
fakehistB  <- py_load_object("fakehistB.npy") %>% drop	
fakercpB  <- py_load_object("fakercpB.npy") %>% drop	
lon  <- py_load_object("lon.npy") %>% drop	
lat  <- py_load_object("lat.npy") %>% drop	

qqsetB <- datasetA
for(i in seq.int(dim(qqsetB)[2])){
  for(j in seq.int(dim(qqsetB)[3])){
    qqsetB[, i, j] <- quantile(datasetB[, i, j], probs = ecdf(datasetA[, i, j])(datasetA[, i, j]))
  }
}
qqvalB <- valsetA
for(i in seq.int(dim(qqsetB)[2])){
  for(j in seq.int(dim(qqsetB)[3])){
    qqvalB[, i, j] <- quantile(datasetB[, i, j], probs = ecdf(datasetA[, i, j])(valsetA[, i, j]))
  }
}
qqhistB <- histsetA
for(i in seq.int(dim(qqsetB)[2])){
  for(j in seq.int(dim(qqsetB)[3])){
    qqhistB[, i, j] <- quantile(datasetB[, i, j], probs = ecdf(datasetA[, i, j])(histsetA[, i, j]))
  }
}
qqrcpB <- rcpsetA
for(i in seq.int(dim(qqsetB)[2])){
  for(j in seq.int(dim(qqsetB)[3])){
    qqrcpB[, i, j] <- quantile(datasetB[, i, j], probs = ecdf(datasetA[, i, j])(rcpsetA[, i, j]))
  }
}
cdftsetB <- datasetA
for(i in seq.int(dim(cdftsetB)[2])){
  for(j in seq.int(dim(cdftsetB)[3])){
    cdftsetB[, i, j] <- CDFt(datasetB[, i, j], datasetA[, i, j], datasetA[, i, j], npas = 1000)$DS
  }
}
cdftvalB <- valsetA
for(i in seq.int(dim(cdftsetB)[2])){
  for(j in seq.int(dim(cdftsetB)[3])){
    cdftvalB[, i, j] <- CDFt(datasetB[, i, j], datasetA[, i, j], valsetA[, i, j], npas = 1000)$DS
  }
}
cdfthistB <- histsetA
for(i in seq.int(dim(cdftsetB)[2])){
  for(j in seq.int(dim(cdftsetB)[3])){
    cdfthistB[, i, j] <- CDFt(datasetB[, i, j], datasetA[, i, j], histsetA[, i, j], npas = 1000)$DS
  }
}
cdftrcpB <- rcpsetA
for(i in seq.int(dim(cdftsetB)[2])){
  for(j in seq.int(dim(cdftsetB)[3])){
    cdftrcpB[, i, j] <- CDFt(datasetB[, i, j], datasetA[, i, j], rcpsetA[, i, j], npas = 1000)$DS
  }
}
print("***********************************************")
print("Scores on calibration set")
print("***********************************************")
sprintf('bias genA2B: %f', mean((fakesetB - datasetB)))
sprintf('bias cdftA2B : %f', mean((cdftsetB - datasetB)))
sprintf('bias qqA2B : %f', mean((qqsetB - datasetB)))
sprintf('bias base A: %f', mean((datasetA - datasetB)))
sprintf('rmse genA2B: %f', mean((fakesetB - datasetB)^2) %>% sqrt)
sprintf('rmse cdftA2B : %f', mean((cdftsetB - datasetB)^2) %>% sqrt)
sprintf('rmse qqA2B : %f', mean((qqsetB - datasetB)^2) %>% sqrt)
sprintf('rmse base A: %f', mean((datasetA - datasetB)^2) %>% sqrt)
sprintf('corr genA2B: %f', cor(c(fakesetB), c(datasetB)))
sprintf('corr cdftA2B : %f', cor(c(cdftsetB), c(datasetB)))
sprintf('corr qqA2B : %f', cor(c(qqsetB), c(datasetB)))
sprintf('corr baseA: %f', cor(c(datasetA), c(datasetB)))

print("***********************************************")
print("Scores on validation set")
print("***********************************************")
sprintf('bias genA2B: %f', mean((fakevalB - valsetB)))
sprintf('bias cdftA2B : %f', mean((cdftvalB - valsetB)))
sprintf('bias qqA2B : %f', mean((qqvalB - valsetB)))
sprintf('bias base A: %f', mean((valsetA - valsetB)))
sprintf('rmse genA2B: %f', mean((fakevalB - valsetB)^2) %>% sqrt)
sprintf('rmse cdftA2B : %f', mean((cdftvalB - valsetB)^2) %>% sqrt)
sprintf('rmse qqA2B : %f', mean((qqvalB - valsetB)^2) %>% sqrt)
sprintf('rmse base A: %f', mean((valsetA - valsetB)^2) %>% sqrt)
sprintf('corr genA2B: %f', cor(c(fakevalB), c(valsetB)))
sprintf('corr cdftA2B : %f', cor(c(cdftvalB), c(valsetB)))
sprintf('corr qqA2B : %f', cor(c(qqvalB), c(valsetB)))
sprintf('corr baseA: %f', cor(c(valsetA), c(valsetB)))

print("***********************************************")
print("Scores on historical set")
print("***********************************************")
sprintf('bias genA2B: %f', mean((fakehistB - histsetB)))
sprintf('bias cdftA2B : %f', mean((cdfthistB - histsetB)))
sprintf('bias qqA2B : %f', mean((qqhistB - histsetB)))
sprintf('bias base A: %f', mean((histsetA - histsetB)))
sprintf('rmse genA2B: %f', mean((fakehistB - histsetB)^2) %>% sqrt)
sprintf('rmse cdftA2B : %f', mean((cdfthistB - histsetB)^2) %>% sqrt)
sprintf('rmse qqA2B : %f', mean((qqhistB - histsetB)^2) %>% sqrt)
sprintf('rmse base A: %f', mean((histsetA - histsetB)^2) %>% sqrt)
sprintf('corr genA2B: %f', cor(c(fakehistB), c(histsetB)))
sprintf('corr cdftA2B : %f', cor(c(cdfthistB), c(histsetB)))
sprintf('corr qqA2B : %f', cor(c(qqhistB), c(histsetB)))
sprintf('corr baseA: %f', cor(c(histsetA), c(histsetB)))

print("***********************************************")
print("Scores on rcp set")
print("***********************************************")
sprintf('bias genA2B: %f', mean((fakercpB - rcpsetB)))
sprintf('bias cdftA2B : %f', mean((cdftrcpB - rcpsetB)))
sprintf('bias qqA2B : %f', mean((qqrcpB - rcpsetB)))
sprintf('bias base A: %f', mean((rcpsetA - rcpsetB)))
sprintf('rmse genA2B: %f', mean((fakercpB - rcpsetB)^2) %>% sqrt)
sprintf('rmse cdftA2B : %f', mean((cdftrcpB - rcpsetB)^2) %>% sqrt)
sprintf('rmse qqA2B : %f', mean((qqrcpB - rcpsetB)^2) %>% sqrt)
sprintf('rmse base A: %f', mean((rcpsetA - rcpsetB)^2) %>% sqrt)
sprintf('corr genA2B: %f', cor(c(fakercpB), c(rcpsetB)))
sprintf('corr cdftA2B : %f', cor(c(cdftrcpB), c(rcpsetB)))
sprintf('corr qqA2B : %f', cor(c(qqrcpB), c(rcpsetB)))
sprintf('corr baseA: %f', cor(c(rcpsetA), c(rcpsetB)))

paltas  <- colorRampPalette(brewer.pal(9, name = "YlGnBu"))(31)
paldiff  <- colorRampPalette(rev(brewer.pal(11, name = "RdYlBu")))(31)

meanA  <- t(apply(datasetA, 2:3, mean))
meanqqB  <- t(apply(qqsetB, 2:3, mean))
meancdftB  <- t(apply(cdftsetB, 2:3, mean))
meanfakeB  <- t(apply(fakesetB, 2:3, mean))
meanB  <- t(apply(datasetB, 2:3, mean))
zlim  <- range(meanA, meanqqB, meanfakeB, meanB)
png("mean_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor * 720)
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, meanA, zlim = zlim, col = paltas, main = "mean A")
lines(world) 
image.plot(lon, lat, meancdftB, zlim = zlim, col = paltas, main = "mean cdftB")
lines(world) 
image.plot(lon, lat, meanfakeB, zlim = zlim, col = paltas, main = "mean ganB")
lines(world) 
image.plot(lon, lat, meanB, zlim = zlim, col = paltas, main = "mean B")
lines(world) 
dev.off()

png("diffmean_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor *  720)
zlim  <- range(c(meanA, meancdftB, meanfakeB) - c(meanB)) %>% abs %>% max %>% "*"(., c(-1, 1))
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, meanA - meanB, zlim = zlim, col = paldiff, main = "mean A-B")
lines(world) 
image.plot(lon, lat, meancdftB - meanB , zlim = zlim, col = paldiff, main = "mean cdftB-B")
lines(world) 
image.plot(lon, lat, meanfakeB - meanB, zlim = zlim, col = paldiff, main = "mean ganB-B")
lines(world) 
image.plot(lon, lat, meanB - meanB, zlim = zlim, col = paldiff, main = "mean B-B")
lines(world) 
dev.off()

sdA  <- t(apply(datasetA, 2:3, sd))
sdcdftB  <- t(apply(cdftsetB, 2:3, sd))
sdfakeB  <- t(apply(fakesetB, 2:3, sd))
sdB  <- t(apply(datasetB, 2:3, sd))
zlim  <- range(sdA, sdcdftB, sdfakeB, sdA)
png("sd_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor * 720)
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, sdA, zlim = zlim, col = paltas, main = "sd A")
lines(world) 
image.plot(lon, lat, sdcdftB, zlim = zlim, col = paltas, main = "sd cdftB")
lines(world) 
image.plot(lon, lat, sdfakeB, zlim = zlim, col = paltas, main = "sd ganB")
lines(world) 
image.plot(lon, lat, sdB, zlim = zlim, col = paltas, main = "sd B")
lines(world) 
dev.off()

zlim  <- range(c(sdA, sdcdftB, sdfakeB) - c(sdB)) %>% abs %>% max %>% "*"(., c(-1, 1))
png("diffsd_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor * 720)
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, sdA - sdB, zlim = zlim, col = paldiff, main = "sd A - sd B")
lines(world) 
image.plot(lon, lat, sdcdftB - sdB , zlim = zlim, col = paldiff, main = "sd cdftB - sd B")
lines(world) 
image.plot(lon, lat, sdfakeB - sdB, zlim = zlim, col = paldiff, main = "sd ganB - sd B")
lines(world) 
image.plot(lon, lat, sdB - sdB, zlim = zlim, col = paldiff, main = "sd B - sd B")
lines(world) 
dev.off()

minA  <- t(apply(datasetA, 2:3, min))
mincdftB  <- t(apply(cdftsetB, 2:3, min))
minfakeB  <- t(apply(fakesetB, 2:3, min))
minB  <- t(apply(datasetB, 2:3, min))
zlim  <- range(minA, mincdftB, minfakeB, minA)
png("min_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor * 720)
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, minA, zlim = zlim, col = paltas, main = "min A")
lines(world) 
image.plot(lon, lat, mincdftB, zlim = zlim, col = paltas, main = "min cdftB")
lines(world) 
image.plot(lon, lat, minfakeB, zlim = zlim, col = paltas, main = "min ganB")
lines(world) 
image.plot(lon, lat, minB, zlim = zlim, col = paltas, main = "min B")
lines(world) 
dev.off()

zlim  <- range(c(minA, mincdftB, minfakeB) - c(minB)) %>% abs %>% max %>% "*"(., c(-1, 1))
png("diffmin_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor * 720)
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, minA - minB, zlim = zlim, col = paldiff, main = "min A - min B")
lines(world) 
image.plot(lon, lat, mincdftB - minB , zlim = zlim, col = paldiff, main = "min cdftB - min B")
lines(world) 
image.plot(lon, lat, minfakeB - minB, zlim = zlim, col = paldiff, main = "min ganB - min B")
lines(world) 
image.plot(lon, lat, minB - minB, zlim = zlim, col = paldiff, main = "min B - min B")
lines(world) 
dev.off()

maxA  <- t(apply(datasetA, 2:3, max))
maxcdftB  <- t(apply(cdftsetB, 2:3, max))
maxfakeB  <- t(apply(fakesetB, 2:3, max))
maxB  <- t(apply(datasetB, 2:3, max))
zlim  <- range(maxA, maxcdftB, maxfakeB, maxA)
png("max_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor * 720)
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, maxA, zlim = zlim, col = paltas, main = "max A")
lines(world) 
image.plot(lon, lat, maxcdftB, zlim = zlim, col = paltas, main = "max cdftB")
lines(world) 
image.plot(lon, lat, maxfakeB, zlim = zlim, col = paltas, main = "max ganB")
lines(world) 
image.plot(lon, lat, maxB, zlim = zlim, col = paltas, main = "max B")
lines(world) 
dev.off()

zlim  <- range(c(maxA, maxcdftB, maxfakeB) - c(maxB)) %>% abs %>% max %>% "*"(., c(-1, 1))
png("diffmax_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor * 720)
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, maxA - maxB, zlim = zlim, col = paldiff, main = "max A - max B")
lines(world) 
image.plot(lon, lat, maxcdftB - maxB , zlim = zlim, col = paldiff, main = "max cdftB - max B")
lines(world) 
image.plot(lon, lat, maxfakeB - maxB, zlim = zlim, col = paldiff, main = "max ganB - max B")
lines(world) 
image.plot(lon, lat, maxB - maxB, zlim = zlim, col = paldiff, main = "max B - max B")
lines(world) 
dev.off()

ar1A  <- t(apply(datasetA, 2:3, function(x) acf(x, lag.max = 1)$acf[2]))
ar1cdftB  <- t(apply(cdftsetB, 2:3, function(x) acf(x, lag.max = 1)$acf[2]))
ar1fakeB  <- t(apply(fakesetB, 2:3, function(x) acf(x, lag.max = 1)$acf[2]))
ar1B  <- t(apply(datasetB, 2:3, function(x) acf(x, lag.max = 1)$acf[2]))
zlim  <- range(ar1A, ar1cdftB, ar1fakeB, ar1A)
png("ar1_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor * 720)
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, ar1A, zlim = zlim, col = paltas, main = "ar1 A")
lines(world) 
image.plot(lon, lat, ar1cdftB, zlim = zlim, col = paltas, main = "ar1 cdftB")
lines(world) 
image.plot(lon, lat, ar1fakeB, zlim = zlim, col = paltas, main = "ar1 ganB")
lines(world) 
image.plot(lon, lat, ar1B, zlim = zlim, col = paltas, main = "ar1 B")
lines(world) 
dev.off()

zlim  <- range(c(ar1A, ar1cdftB, ar1fakeB) - c(ar1B)) %>% abs %>% max %>% "*"(., c(-1, 1))
png("diffar1_cal.png", res = 72*resfactor, width = resfactor * 720, height = resfactor * 720)
par(mfrow = c(2, 2))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, ar1A - ar1B, zlim = zlim, col = paldiff, main = "ar1 A - ar1 B")
lines(world) 
image.plot(lon, lat, ar1cdftB - ar1B , zlim = zlim, col = paldiff, main = "ar1 cdftB - ar1 B")
lines(world) 
image.plot(lon, lat, ar1fakeB - ar1B, zlim = zlim, col = paldiff, main = "ar1 ganB - ar1 B")
lines(world) 
image.plot(lon, lat, ar1B - ar1B, zlim = zlim, col = paldiff, main = "ar1 B - ar1 B")
lines(world) 
dev.off()


spatial_rmse_fakercpB  <- apply( (fakercpB - rcpsetB)^2, 1, function(x)sqrt(mean(x)))
spatial_rmse_cdftrcpB  <- apply( (cdftrcpB - rcpsetB)^2, 1, function(x)sqrt(mean(x)))
spatial_rmse_qqrcpB  <- apply( (qqrcpB - rcpsetB)^2, 1, function(x)sqrt(mean(x)))
ylim  <- range(spatial_rmse_qqrcpB, spatial_rmse_cdftrcpB, spatial_rmse_fakercpB) 
png("spatial_rmse.png", res = 72*resfactor, width = resfactor * 1280, height = resfactor * 480)
par(mfrow = c(1, 3))
par(mar=c(5, 5, 5, 8))
plot(spatial_rmse_fakercpB, pch = 19, ylim = ylim, main = "spatial: rmse gan B")
plot(spatial_rmse_qqrcpB, pch = 19, ylim = ylim, main = "spatial: rmse qq B")
plot(spatial_rmse_cdftrcpB, pch = 19, ylim = ylim, main = "spatial: rmse cdft B")
# plot(spatial_rmse_fakercpB - spatial_rmse_qqrcpB, pch = 19, main = "spatial: rmse gan B -  rmse qq B")
dev.off()

temporal_rmse_fakercpB  <- apply( (fakercpB - rcpsetB)^2, 2:3, function(x)sqrt(mean(x)))
temporal_rmse_cdftrcpB  <- apply( (cdftrcpB - rcpsetB)^2, 2:3, function(x)sqrt(mean(x)))
temporal_rmse_qqrcpB  <- apply( (qqrcpB - rcpsetB)^2, 2:3, function(x)sqrt(mean(x)))

zlim_diff  <- range(temporal_rmse_qqrcpB - temporal_rmse_fakercpB) %>% abs %>% max %>% "*"(., c(-1, 1))
zlim  <- range(temporal_rmse_qqrcpB, temporal_rmse_fakercpB) 
png("temporal_rmse.png", res = 72*resfactor, width = resfactor * 1280, height = resfactor * 480)
par(mfrow = c(1, 3))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, temporal_rmse_fakercpB, pch = 20, zlim = zlim, col = paltas, main = "temporal: rmse ganB")
lines(world)
image.plot(lon, lat, temporal_rmse_cdftrcpB, pch = 20, zlim = zlim, col = paltas, main = "temporal: rmse cdftB")
lines(world)
image.plot(lon, lat, temporal_rmse_qqrcpB, pch = 20, zlim = zlim, col = paltas, main = "temporal: rmse qqB")
lines(world)
#image.plot(lon, lat, temporal_rmse_fakercpB - temporal_rmse_qqrcpB, pch = 20, zlim = zlim_diff, col = paldiff, main = "temporal: rmse gan B -  rmse qq B")
#lines(world)
dev.off()

spatial_corr_fakercpB  <- fakercpB[, 1, 1]
spatial_corr_cdftrcpB  <- fakercpB[, 1, 1]
spatial_corr_qqrcpB  <- fakercpB[, 1, 1]
for(i in 1:nrow(fakercpB)){
  spatial_corr_fakercpB[i] = cor(c(fakercpB[i,,]), c(rcpsetB[i,,]))
  spatial_corr_cdftrcpB[i] = cor(c(cdftrcpB[i,,]), c(rcpsetB[i,,]))
  spatial_corr_qqrcpB[i] = cor(c(qqrcpB[i,,]), c(rcpsetB[i,,]))
}
ylim  <- range(spatial_corr_qqrcpB, spatial_corr_qqrcpB, spatial_corr_fakercpB) 
png("spatial_corr.png", res = 72*resfactor, width = resfactor * 1280, height = resfactor * 480)
par(mfrow = c(1, 3))
par(mar=c(5, 5, 5, 8))
plot(spatial_corr_fakercpB, pch = 19, ylim = ylim, main = "spatial: corr gan B")
plot(spatial_corr_cdftrcpB, pch = 19, ylim = ylim, main = "spatial: corr cdft B")
plot(spatial_corr_qqrcpB, pch = 19, ylim = ylim, main = "spatial: corr qq B")
# plot(spatial_corr_fakercpB - spatial_corr_qqrcpB, pch = 19, main = "spatial: corr gan B -  corr qqB")
dev.off()

temporal_corr_fakercpB  <- fakercpB[1, , ]
temporal_corr_cdftrcpB  <- fakercpB[1, , ]
temporal_corr_qqrcpB  <- fakercpB[1, , ]
for(i in 1:nrow(temporal_corr_qqrcpB)){
  for(j in 1:ncol(temporal_corr_qqrcpB)){
    temporal_corr_fakercpB[i, j] = cor(fakercpB[,i,j], rcpsetB[,i,j])
    temporal_corr_cdftrcpB[i, j] = cor(cdftrcpB[,i,j], rcpsetB[,i,j])
    temporal_corr_qqrcpB[i, j] = cor(qqrcpB[,i,j], rcpsetB[,i,j])
  }
}

zlim_diff  <- range(temporal_corr_qqrcpB - temporal_corr_fakercpB) %>% abs %>% max %>% "*"(., c(-1, 1))
zlim  <- range(temporal_corr_qqrcpB, temporal_corr_cdftrcpB, temporal_corr_fakercpB) 
png("temporal_corr.png", res = 72*resfactor, width = resfactor * 1280, height = resfactor * 480)
par(mfrow = c(1, 3))
par(mar=c(5, 5, 5, 8))
image.plot(lon, lat, temporal_corr_fakercpB, pch = 20, zlim = zlim, col = paltas, main = "temporal: corr ganB")
lines(world)
image.plot(lon, lat, temporal_corr_cdftrcpB, pch = 20, zlim = zlim, col = paltas, main = "temporal: corr cdftB")
lines(world)
image.plot(lon, lat, temporal_corr_qqrcpB, pch = 20, zlim = zlim, col = paltas, main = "temporal: corr qqB")
lines(world)
# image.plot(lon, lat, temporal_corr_fakercpB - temporal_corr_qqrcpB, pch = 20, zlim = zlim_diff, col = paldiff, main = "temporal: corr gan B -  corr qqB")
# lines(world)
dev.off()

# zlim  <- range(c(qqsetB, fakesetB) - c(datasetB)) %>% abs %>% max %>% "*"(., c(-1, 1))
# for (i in 1:nrow(datasetA)) {
#   filename = sprintf("img/cal%05d.png", i )
#   png(filename, width = 1280, height = 480)
# draw your plots here, then pause for a while with
#   A  <- t(datasetA[i,,])
#   qqB  <- t(qqsetB[i,,])
#   fakeB  <- t(fakesetB[i,,])
#   B  <- t(datasetB[i,,])
#   par(mfrow = c(1, 3))
#   image.plot(lon, lat, B, zlim = c(0, 1), main = "B", col = paltas)
#   lines(world) 
#   image.plot(lon, lat, qqB - B, zlim = zlim, main = "qqB - B", col = paldiff)
#   lines(world) 
#   image.plot(lon, lat, fakeB - B, zlim = zlim, main = "ganB - B", col = paldiff)
#   lines(world) 
#   dev.off()
# }
# ffmpeg -i img/val%05d.png -b:v 8000k   -c:v libx264  -s:v 1280x720  video.mp4
# 
# zlim  <- range(c(qqvalB, fakevalB) - c(valsetB)) %>% abs %>% max %>% "*"(., c(-1, 1))
# for (i in 1:nrow(valsetA)) {
#   filename = sprintf("img/val%05d.png", i )
#   png(filename, width = 1280, height = 480)
# draw your plots here, then pause for a while with
#   A  <- t(valsetA[i,,])
#   qqB  <- t(qqvalB[i,,])
#   fakeB  <- t(fakevalB[i,,])
#   B  <- t(valsetB[i,,])
#   par(mfrow = c(1, 3))
#   image.plot(lon, lat, B, zlim = c(0, 1), main = "B", col = paltas)
#   lines(world) 
#   image.plot(lon, lat, qqB - B, zlim = zlim, main = "qqB - B", col = paldiff)
#   lines(world) 
#   image.plot(lon, lat, fakeB - B, zlim = zlim, main = "ganB - B", col = paldiff)
#   lines(world) 
#   dev.off()
# }
# 
# zlim  <- range(c(qqhistB, fakehistB) - c(histsetB)) %>% abs %>% max %>% "*"(., c(-1, 1))
# for (i in 1:nrow(histsetA)) {
#   filename = sprintf("img/hist%05d.png", i )
#   png(filename, width = 1280, height = 480)
# draw your plots here, then pause for a while with
#   A  <- t(histsetA[i,,])
#   qqB  <- t(qqhistB[i,,])
#   fakeB  <- t(fakehistB[i,,])
#   B  <- t(histsetB[i,,])
#   par(mfrow = c(1, 3))
# image.plot(lon, lat, A - B, zlim = zlim, main = "A - B", col = paldiff)
# lines(world) 
#   image.plot(lon, lat, B, zlim = c(0, 1), main = "B", col = paltas)
#   lines(world) 
#   image.plot(lon, lat, qqB - B, zlim = zlim, main = "qqB - B", col = paldiff)
#   lines(world) 
#   image.plot(lon, lat, fakeB - B, zlim = zlim, main = "ganB - B", col = paldiff)
#   lines(world) 
#   dev.off()
# }
# 
zlim  <- range(c(qqrcpB, cdftrcpB, fakercpB) - c(rcpsetB)) %>% abs %>% max %>% "*"(., c(-1, 1))
for (i in 1:nrow(rcpsetA)) {
  filename = sprintf("img/rcp%05d.png", i )
  png(filename, width = 960, height = 960)
  A  <- t(rcpsetA[i,,])
  qqB  <- t(qqrcpB[i,,])
  cdftB  <- t(cdftrcpB[i,,])
  fakeB  <- t(fakercpB[i,,])
  B  <- t(rcpsetB[i,,])
  par(mfrow = c(2, 2))
  image.plot(lon, lat, B, zlim = c(0, 1), main = "B", col = paltas)
  lines(world) 
  image.plot(lon, lat, qqB - B, zlim = zlim, main = "qqB - B", col = paldiff)
  lines(world) 
  image.plot(lon, lat, cdftB - B, zlim = zlim, main = "cdftB - B", col = paldiff)
  lines(world) 
  image.plot(lon, lat, fakeB - B, zlim = zlim, main = "ganB - B", col = paldiff)
  lines(world) 
  dev.off()
}
