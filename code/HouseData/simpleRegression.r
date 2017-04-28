library(MASS)

# install.packages("klaR")
# install.packages("caret")

library("klaR")
library("caret")
library("e1071")
library("glmnet")

library(aod)
library(ggplot2)
library(Rcpp)
library(np)

library(elasticnet)

datasetFolder = "../../datasets/"
housedatasetFolder = paste(datasetFolder,"HouseData/",sep="")

testFile = paste(housedatasetFolder,"testCleanSparse_extra.csv",sep="")
trainFile = paste(housedatasetFolder,"trainCleanSparse_extra.csv",sep="")

testFileNotSparse = paste(housedatasetFolder,"testClean.csv",sep="")
trainFileNotSparse = paste(housedatasetFolder,"trainClean.csv",sep="")

trainData = read.csv(trainFile)
testData = read.csv(testFile)

trainDataNotSparse = read.csv(trainFileNotSparse)
testDataNotSparse = read.csv(testFileNotSparse)

trainData[is.na(trainData)] <- 0
testData[is.na(testData)] <- 0

trainDataNotSparse[is.na(trainDataNotSparse)] <- 0
testDataNotSparse[is.na(testDataNotSparse)] <- 0

errorinTrainLinearModel = 0
errorinTestLinearModel = 0
errorinTrainLinearModelNotSparse = 0
errorinTestLinearModelNotSparse = 0

errorinTrainLogit = 0
errorinTestLogit = 0

errorinTrainENet = 0
errorinTestENet = 0

errorinTrainl = 0
errorinTestl = 0

errorinTestPCA = rep(0,250)

numRuns = 20

for(run in 1:numRuns){
	print(run)
	randomTrainData = trainData[sample(nrow(trainData)),]
	randomTrainDataNotSparse = trainDataNotSparse[sample(nrow(trainDataNotSparse)),]
	
	logitTrainData = randomTrainData

	# maxPrice = max(logitTrainData[,length(logitTrainData[1,])])
	maxPrice = 1000000

	logitTrainData[,length(logitTrainData[1,])] = logitTrainData[,length(logitTrainData[1,])]/maxPrice

	logitTrainData_Train = logitTrainData[1:1200,]
	logitTrainData_Test = logitTrainData[1201:length(logitTrainData[,1]),]

	trainData_Train = randomTrainData[1:1200,]
	trainData_Test = randomTrainData[1201:length(trainData[,1]),]

	trainDataNotSparse_Train = randomTrainDataNotSparse[1:1200,]
	trainDataNotSparse_Test = randomTrainDataNotSparse[1201:length(randomTrainDataNotSparse[,1]),]

	# # mod = lm(SalePrice ~ ., trainData_Train[,2:length(trainData_Train[1,])])
	# errorinTrainLinearModel = errorinTrainLinearModel + sum(abs((array(mod$fitted.values) - trainData_Train[,length(trainData_Train)]))/trainData_Train[,length(trainData_Train)])/length(trainData_Train[,1])
	# 
	# testPredictions = predict(mod, trainData_Test[,2:length(trainData_Test[1,])])
	# errorinTestLinearModel = errorinTestLinearModel + sum(abs((array(testPredictions) - trainData_Test[,length(trainData_Test)]))/trainData_Test[,length(trainData_Test)])/length(trainData_Test[,1])
	mod = lm(SalePrice ~ ., logitTrainData_Train[,2:length(logitTrainData_Train[1,])])
	errorinTrainLinearModel = errorinTrainLinearModel + sum(abs((array(mod$fitted.values) - logitTrainData_Train[,length(logitTrainData_Train)]))/logitTrainData_Train[,length(logitTrainData_Train)])/length(logitTrainData_Train[,1])

	testPredictions = predict(mod, logitTrainData_Test[,2:length(logitTrainData_Test[1,])])
	errorinTestLinearModel = errorinTestLinearModel + sum(abs((array(testPredictions) - logitTrainData_Test[,length(logitTrainData_Test)]))/logitTrainData_Test[,length(logitTrainData_Test)])/length(logitTrainData_Test[,1])
	
	print("Linear model Sparse:")
	# print(paste("Train error: ",errorinTrain,"Tresting Error:",errorinTest))

	# notSparse.mod = lm(SalePrice ~ ., data = trainDataNotSparse_Train[,2:length(trainDataNotSparse_Train[1,])])
	# 
	# errorinTrainLinearModelNotSparse = errorinTrainLinearModelNotSparse + sum(abs((array(notSparse.mod$fitted.values) - trainDataNotSparse_Train[,length(trainDataNotSparse_Train)]))/trainDataNotSparse_Train[,length(trainDataNotSparse_Train)])/length(trainDataNotSparse_Train[,1])
	# 
	# testPredictions = predict(notSparse.mod, trainDataNotSparse_Test[,2:length(trainDataNotSparse_Test[1,])])
	# errorinTestLinearModelNotSparse = errorinTestLinearModelNotSparse + sum(abs((array(testPredictions) - trainDataNotSparse_Test[,length(trainDataNotSparse_Test)]))/trainDataNotSparse_Test[,length(trainDataNotSparse_Test)])/length(trainDataNotSparse_Test[,1])

	print("Linear model Not Sparse:")
	# print(paste("Train error: ",errorinTrain,"Tresting Error:",errorinTest))


	modLogit = glm(SalePrice ~ ., data = logitTrainData_Train[,2:length(logitTrainData_Train[1,])], family = "binomial")


	errorinTrainLogit = errorinTrainLogit + sum(abs((array(modLogit$fitted.values) - logitTrainData_Train[,length(logitTrainData_Train)]))/logitTrainData_Train[,length(logitTrainData_Train)])/length(logitTrainData_Train[,1])

	testPredictions = predict(modLogit, logitTrainData_Test[,2:length(trainData_Test[1,])])
	errorinTestLogit = errorinTestLogit + sum(abs((array(testPredictions) - logitTrainData_Test[,ncol(logitTrainData_Test)]))/logitTrainData_Test[,ncol(logitTrainData_Test)])/nrow(logitTrainData_Test[,1])

	print("Logistic model:")
	# print(paste("Train error: ",errorinTrain,"Tresting Error:",errorinTest))

	# fitnet = glmnet(x = as.matrix(trainData_Train[,2:(length(trainData_Train[1,])-1)]), y = as.matrix(trainData_Train[,length(trainData_Train[1,])]), alpha = 1)
	# cv.fitnet = cv.glmnet(x = as.matrix(trainData_Train[,2:(length(trainData_Train[1,])-1)]), y = as.matrix(trainData_Train[,length(trainData_Train[1,])]), alpha = 1)
	# 
	# 
	# testPredictions = predict(fitnet, as.matrix(trainData_Train[,2:(length(trainData_Train[1,])-1)]))
	# errorinTrainENet = errorinTrainENet + sum(abs((array(testPredictions) - trainData_Train[,length(trainData_Train)]))/trainData_Train[,length(trainData_Train)])/length(trainData_Train[,1])
	# mte=apply((testPredictions-trainData_Train[,length(trainData_Train)])^2,2,mean)
	# 
	# 
	# testPredictions = predict(fitnet, as.matrix(trainData_Test[,2:(length(trainData_Test[1,])-1)]))
	# errorinTestENet = errorinTestENet + sum(abs((array(testPredictions) - trainData_Test[,length(trainData_Test)]))/trainData_Test[,length(trainData_Test)])/length(trainData_Test[,1])

	print("elasticnet:")
	# print(paste("Train error: ",errorinTrain,"Tresting Error:",errorinTest))
	

	# ridge.mod = lm.ridge(SalePrice ~ ., data = trainDataNotSparse_Train[,2:length(trainDataNotSparse_Train[1,])])
	# n <- names(trainDataNotSparse_Train[,2:length(trainDataNotSparse_Train[1,])])
	# f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))

	# kern.mod = ksmooth(x = as.matrix(trainDataNotSparse_Train[,2:(length(trainDataNotSparse_Train[1,])-1)]), y = as.matrix(trainDataNotSparse_Train[,length(trainDataNotSparse_Train[1,])]))
	# kern.mod = npreg(formula = f, data = trainDataNotSparse_Train[,2:length(trainDataNotSparse_Train[1,])])

	# trainPred = predict(kern.mod, trainDataNotSparse_Train[,2:(length(trainDataNotSparse_Train[1,])-1)])
	# errorinTrain = sum(abs((array(kern.mod$ym) - trainDataNotSparse_Train[,length(trainDataNotSparse_Train)]))/trainDataNotSparse_Train[,length(trainDataNotSparse_Train)])/length(trainDataNotSparse_Train[,1])
	# # 
	# testPredictions = predict(mod, trainDataNotSparse_Test[,2:length(trainDataNotSparse_Test[1,])])
	# errorinTest = sum(abs((array(testPredictions) - trainDataNotSparse_Test[,length(trainDataNotSparse_Test)]))/trainDataNotSparse_Test[,length(trainDataNotSparse_Test)])/length(trainDataNotSparse_Test[,1])


	print("Ridge Regression:")



	# enetMod = enet(x = as.matrix(trainData_Train[,2:length(trainData_Train[1,])-1]), y = as.matrix(trainData_Train[,length(trainData_Train[1,])]))


	#
	# library("neuralnet")
	#
	# n <- names(trainData_Train[,2:length(trainData_Train[1,])])
	# f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))
	# 
	# net = neuralnet(f, data = trainData_Train[,2:length(trainData_Train[1,])], hidden = 10, threshold = 0.1, stepmax = 1e10)
	# 
	# preds = compute(net,trainData_Test[,2:length(trainData_Test[1,])-1])
	# errorinTest = sum(abs((array(preds) - trainData_Test[,length(trainData_Test)]))/trainData_Test[,length(trainData_Test)])/length(trainData_Test[,1])
	# 

	# numCenters = 25
	# 
	# cluster = kmeans(trainData[,33:57], centers = numCenters)
	#
	# mods <- vector(mode = "list", length = numCenters)
	# 
	# 
	# for(i in 1:numCenters){
	# 	mods[[i]] = lm(SalePrice ~ ., trainData_Train[(cluster$cluster[1:1200]==i),2:length(trainData_Train[1,])])
	# 	e = sum(abs((array(mods[[i]]$fitted.values) - trainData_Train[cluster$cluster[1:1200]==i,length(trainData_Train)]))/trainData_Train[cluster$cluster[1:1200]==i,length(trainData_Train[1,])])/length(trainData_Train[cluster$cluster[1:1200]==i,1])
	# 	if(is.na(e) || is.nan(e)){ e = 0 }
	# 	errorinTrainl[i] = errorinTrainl[i] + e
	# }
	# 
	# for(i in 1:numCenters){
	# 	# print(1)
	# 	testPredictions = predict(mods[[i]], trainData_Test[cluster$cluster[1201:length(cluster$cluster)]==i,2:length(trainData_Test[1,])])
	# 	# print(testPredictions)
	# 	# print(2)
	# 	e = sum(abs((array(testPredictions) - trainData_Test[cluster$cluster[1201:length(cluster$cluster)]==i,length(trainData_Test)]))/trainData_Test[cluster$cluster[1201:length(cluster$cluster)]==i,length(trainData_Test)])/length(trainData_Test[cluster$cluster[1201:length(cluster$cluster)]==i,1])
	# 	if(is.na(e) || is.nan(e)){ e = 0 }
	# 	errorinTestl[i] = errorinTestl[i] + e
	# 	# print(3)
	# 
	# 	# print(1)
	# 	# testPredictions = predict(mods[[i]], trainData[cluster$cluster[1:length(cluster$cluster)]==i,2:length(trainData[1,])])
	# 	# print(testPredictions)
	# 	# print(2)
	# 	# errorinTestl[i] = sum(abs((array(testPredictions) - trainData[cluster$cluster[1:length(cluster$cluster)]==i,length(trainData)]))/trainData[cluster$cluster[1:length(cluster$cluster)]==i,length(trainData)])/length(trainData[cluster$cluster[1:length(cluster$cluster)]==i,1])
	# 	# print(3)
	# }
	# 
	# errorinTestl[errorinTestl == 'NaN'] = 0







	# n <- names(trainData_Train[,2:length(trainData_Train[1,])])
	# f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))
	# 
	# 
	# # #
	data.pca = prcomp(trainData_Train[,2:(length(trainData_Train[1,])-1)])
	newTrainData = data.frame(SalePrice = trainData_Train[,"SalePrice"]/maxPrice, data.pca$x)
	
	print("PCA")
	f = "SalePrice ~ PC1"
	form = as.formula(f)
	pcamod = glm(form, data = newTrainData, family = gaussian)
	# print("Predict")
	test.data = predict(data.pca, newdata=trainData_Test[,2:(length(trainData_Test[1,])-1)])
	# print("Predict2")
	pred <- predict(pcamod, newdata = data.frame(test.data), type = "response")
	errorinTest = sum(abs((array(pred) - trainData_Test[,length(trainData_Test)]/maxPrice))/(trainData_Test[,length(trainData_Test)]/maxPrice))/length(trainData_Test[,1])
	# print(paste("PCA with principle components 1 through ",1))
	# print(paste("Test Error: ",errorinTest))
	errorinTestPCA[1] = errorinTestPCA[1] + errorinTest
	for(i in 2:250){
		f = paste(f, paste("PC", i,sep=""), sep= " + ")
		form = as.formula(f)
		pcamod = glm(form, data = newTrainData, family = binomial)
		# print("Predict")
		test.data = predict(data.pca, newdata=trainData_Test[,2:(length(trainData_Train[1,])-1)])
		# print("Predict2")
		pred <- predict(pcamod, newdata = data.frame(test.data), type = "response")
		errorinTest = sum(abs((array(pred) - trainData_Test[,length(trainData_Test)]/maxPrice))/(trainData_Test[,length(trainData_Test)]/maxPrice))/length(trainData_Test[,1])
		# print(paste("PCA with principle components 1 through ",i))
		# print(paste("Test Error: ",errorinTest))
		errorinTestPCA[i] = errorinTestPCA[i] + errorinTest
	
	}
	
	plot(errorinTestPCA/run,type="l")



}

print("Linear Model:")
print(paste("Train:",errorinTrainLinearModel/numRuns,"Test:",errorinTestLinearModel/numRuns))

print("Linear Model Not Sparse:")
print(paste("Train:",errorinTrainLinearModelNotSparse/numRuns,"Test:",errorinTestLinearModelNotSparse/numRuns))

print("Logit Model:")
print(paste("Train:",errorinTrainLogit/numRuns,"Test:",errorinTestLogit/numRuns))

print("Enet Model:")
print(paste("Train:",errorinTrainENet/numRuns,"Test:",errorinTestENet/numRuns))

print("PCA Model:")
print(min(errorinTestPCA/numRuns))

plot(errorinTestPCA/numRuns, type="l", xlab = "PCA Features")

# errorinTestPCA = rep(0,243)






# 
# 
# trainData_Train = trainData[1:1200,]
# trainData_Test = trainData[1201:length(trainData[,1]),]
# 
# trainDataNotSparse_Train = trainDataNotSparse[1:1200,]
# trainDataNotSparse_Test = trainDataNotSparse[1201:length(trainDataNotSparse[,1]),]
# 
# mod = lm(SalePrice ~ ., trainData_Train[,2:length(trainData_Train[1,])])
# errorinTrain = sum(abs((array(mod$fitted.values) - trainData_Train[,length(trainData_Train)]))/trainData_Train[,length(trainData_Train)])/length(trainData_Train[,1])
# 
# testPredictions = predict(mod, trainData_Test[,2:length(trainData_Test[1,])])
# errorinTest = sum(abs((array(testPredictions) - trainData_Test[,length(trainData_Test)]))/trainData_Test[,length(trainData_Test)])/length(trainData_Test[,1])
# 
# print("Linear model Sparse:")
# print(paste("Train error: ",errorinTrain,"Tresting Error:",errorinTest))
# 
# notSparse.mod = lm(SalePrice ~ ., data = trainDataNotSparse_Train[,2:length(trainDataNotSparse_Train[1,])])
# 
# errorinTrain = sum(abs((array(notSparse.mod$fitted.values) - trainDataNotSparse_Train[,length(trainDataNotSparse_Train)]))/trainDataNotSparse_Train[,length(trainDataNotSparse_Train)])/length(trainDataNotSparse_Train[,1])
# 
# testPredictions = predict(notSparse.mod, trainDataNotSparse_Test[,2:length(trainDataNotSparse_Test[1,])])
# errorinTest = sum(abs((array(testPredictions) - trainDataNotSparse_Test[,length(trainDataNotSparse_Test)]))/trainDataNotSparse_Test[,length(trainDataNotSparse_Test)])/length(trainDataNotSparse_Test[,1])
# 
# print("Linear model Not Sparse:")
# print(paste("Train error: ",errorinTrain,"Tresting Error:",errorinTest))
# 
# logitTrainData = trainData
# 
# maxPrice = max(logitTrainData[,length(logitTrainData[1,])])
# 
# logitTrainData[,length(logitTrainData[1,])] = logitTrainData[,length(logitTrainData[1,])]/maxPrice
# 
# logitTrainData_Train = logitTrainData[1:1200,]
# logitTrainData_Test = logitTrainData[1201:length(logitTrainData[,1]),]
# 
# 
# modLogit = glm(SalePrice ~ ., data = logitTrainData_Train[,2:length(logitTrainData_Train[1,])], family = "binomial")
# 
# 
# errorinTrain = sum(abs((array(modLogit$fitted.values) - logitTrainData_Train[,length(logitTrainData_Train)]))/logitTrainData_Train[,length(logitTrainData_Train)])/length(logitTrainData_Train[,1])
# 
# testPredictions = predict(modLogit, logitTrainData_Test[,2:length(trainData_Test[1,])])
# errorinTest = sum(abs((array(testPredictions) - logitTrainData_Test[,length(logitTrainData_Test)]))/logitTrainData_Test[,length(logitTrainData_Test)])/length(logitTrainData_Test[,1])
# 
# print("Logistic model:")
# print(paste("Train error: ",errorinTrain,"Tresting Error:",errorinTest))
# 
# fitnet = glmnet(x = as.matrix(trainData_Train[,2:(length(trainData_Train[1,])-1)]), y = as.matrix(trainData_Train[,length(trainData_Train[1,])]), alpha = 1)
# cv.fitnet = cv.glmnet(x = as.matrix(trainData_Train[,2:(length(trainData_Train[1,])-1)]), y = as.matrix(trainData_Train[,length(trainData_Train[1,])]), alpha = 1)
# 
# 
# testPredictions = predict(fitnet, as.matrix(trainData_Train[,2:(length(trainData_Train[1,])-1)]))
# errorinTrain = sum(abs((array(testPredictions) - trainData_Train[,length(trainData_Train)]))/trainData_Train[,length(trainData_Train)])/length(trainData_Train[,1])
# mte=apply((testPredictions-trainData_Train[,length(trainData_Train)])^2,2,mean)
# 
# 
# testPredictions = predict(fitnet, as.matrix(trainData_Test[,2:(length(trainData_Test[1,])-1)]))
# errorinTest = sum(abs((array(testPredictions) - trainData_Test[,length(trainData_Test)]))/trainData_Test[,length(trainData_Test)])/length(trainData_Test[,1])
# 
# print("elasticnet:")
# print(paste("Train error: ",errorinTrain,"Tresting Error:",errorinTest))
# 
# 
# # ridge.mod = lm.ridge(SalePrice ~ ., data = trainDataNotSparse_Train[,2:length(trainDataNotSparse_Train[1,])])
# n <- names(trainDataNotSparse_Train[,2:length(trainDataNotSparse_Train[1,])])
# f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))
# 
# # kern.mod = ksmooth(x = as.matrix(trainDataNotSparse_Train[,2:(length(trainDataNotSparse_Train[1,])-1)]), y = as.matrix(trainDataNotSparse_Train[,length(trainDataNotSparse_Train[1,])]))
# # kern.mod = npreg(formula = f, data = trainDataNotSparse_Train[,2:length(trainDataNotSparse_Train[1,])])
# 
# # trainPred = predict(kern.mod, trainDataNotSparse_Train[,2:(length(trainDataNotSparse_Train[1,])-1)])
# # errorinTrain = sum(abs((array(kern.mod$ym) - trainDataNotSparse_Train[,length(trainDataNotSparse_Train)]))/trainDataNotSparse_Train[,length(trainDataNotSparse_Train)])/length(trainDataNotSparse_Train[,1])
# # # 
# # testPredictions = predict(mod, trainDataNotSparse_Test[,2:length(trainDataNotSparse_Test[1,])])
# # errorinTest = sum(abs((array(testPredictions) - trainDataNotSparse_Test[,length(trainDataNotSparse_Test)]))/trainDataNotSparse_Test[,length(trainDataNotSparse_Test)])/length(trainDataNotSparse_Test[,1])
# 
# 
# print("Ridge Regression:")
# 
# 
# 
# # enetMod = enet(x = as.matrix(trainData_Train[,2:length(trainData_Train[1,])-1]), y = as.matrix(trainData_Train[,length(trainData_Train[1,])]))
# 
# 
# #
# # library("neuralnet")
# #
# # n <- names(trainData_Train[,2:length(trainData_Train[1,])])
# # f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))
# # 
# # net = neuralnet(f, data = trainData_Train[,2:length(trainData_Train[1,])], hidden = 10, threshold = 0.1, stepmax = 1e10)
# # 
# # preds = compute(net,trainData_Test[,2:length(trainData_Test[1,])-1])
# # errorinTest = sum(abs((array(preds) - trainData_Test[,length(trainData_Test)]))/trainData_Test[,length(trainData_Test)])/length(trainData_Test[,1])
# # 
# 
# numCenters = 5
#
# cluster = kmeans(trainData[,33:57], centers = numCenters)
# 
# mods <- vector(mode = "list", length = numCenters)
# 
# errorinTrainl = 0
# errorinTestl = 0
# 
# 
# for(i in 1:numCenters){
# 	mods[[i]] = lm(SalePrice ~ ., trainData_Train[(cluster$cluster[1:1200]==i),2:length(trainData_Train[1,])])
# 	errorinTrainl[i] = sum(abs((array(mods[[i]]$fitted.values) - trainData_Train[cluster$cluster[1:1200]==i,length(trainData_Train)]))/trainData_Train[cluster$cluster[1:1200]==i,length(trainData_Train[1,])])/length(trainData_Train[cluster$cluster[1:1200]==i,1])
# }
# 
# for(i in 1:numCenters){
# 	print(1)
# 	testPredictions = predict(mods[[i]], trainData_Test[cluster$cluster[1201:length(cluster$cluster)]==i,2:length(trainData_Test[1,])])
# 	print(testPredictions)
# 	print(2)
# 	errorinTestl[i] = sum(abs((array(testPredictions) - trainData_Test[cluster$cluster[1201:length(cluster$cluster)]==i,length(trainData_Test)]))/trainData_Test[cluster$cluster[1201:length(cluster$cluster)]==i,length(trainData_Test)])/length(trainData_Test[cluster$cluster[1201:length(cluster$cluster)]==i,1])
# 	print(3)
# 
# 	# print(1)
# 	# testPredictions = predict(mods[[i]], trainData[cluster$cluster[1:length(cluster$cluster)]==i,2:length(trainData[1,])])
# 	# print(testPredictions)
# 	# print(2)
# 	# errorinTestl[i] = sum(abs((array(testPredictions) - trainData[cluster$cluster[1:length(cluster$cluster)]==i,length(trainData)]))/trainData[cluster$cluster[1:length(cluster$cluster)]==i,length(trainData)])/length(trainData[cluster$cluster[1:length(cluster$cluster)]==i,1])
# 	# print(3)
# }
# 
# errorinTestl[errorinTestl == 'NaN'] = 0
# 
# 
# 
# 
# 
# 
# 
# # n <- names(trainData_Train[,2:length(trainData_Train[1,])])
# # f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))
# # 
# # 
# # # #
# # data.pca = prcomp(trainData_Train[,2:(length(trainData_Train[1,])-1)])
# # newTrainData = data.frame(SalePrice = trainData_Train[,"SalePrice"]/maxPrice, data.pca$x)
# # 
# f = "SalePrice ~ PC1"
# form = as.formula(f)
# pcamod = glm(form, data = newTrainData, family = gaussian)
# # print("Predict")
# test.data = predict(data.pca, newdata=trainData_Test[,2:(length(trainData_Test[1,])-1)])
# # print("Predict2")
# pred <- predict(pcamod, newdata = data.frame(test.data), type = "response")
# errorinTest = sum(abs((array(pred) - trainData_Test[,length(trainData_Test)]/maxPrice))/(trainData_Test[,length(trainData_Test)]/maxPrice))/length(trainData_Test[,1])
# print(paste("PCA with principle components 1 through ",1))
# print(paste("Test Error: ",errorinTest))
# errorinTestl = errorinTest
# for(i in 2:243){
# 	f = paste(f, paste("PC", i,sep=""), sep= " + ")
# 	form = as.formula(f)
# 	pcamod = glm(form, data = newTrainData, family = binomial)
# 	# print("Predict")
# 	test.data = predict(data.pca, newdata=trainData_Test[,2:(length(trainData_Train[1,])-1)])
# 	# print("Predict2")
# 	pred <- predict(pcamod, newdata = data.frame(test.data), type = "response")
# 	errorinTest = sum(abs((array(pred) - trainData_Test[,length(trainData_Test)]/maxPrice))/(trainData_Test[,length(trainData_Test)]/maxPrice))/length(trainData_Test[,1])
# 	print(paste("PCA with principle components 1 through ",i))
# 	print(paste("Test Error: ",errorinTest))
# 	errorinTestl[i] = errorinTest
# 
# }
# 
# plot(errorinTestl,type="l")
