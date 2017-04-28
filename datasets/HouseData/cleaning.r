


trainData = read.csv("train_ru.csv", sep=",", stringsAsFactors=FALSE)
testData = read.csv("test_ru.csv", sep=",", stringsAsFactors=FALSE)

trainDataIsNumeric = TRUE

for(i in 1:length(trainData[1,])){
	trainDataIsNumeric[i] = is.numeric(trainData[1,i])
}

write("","dataDescription.txt")

trainDataSparsed = matrix(0, length(trainData[,1]),0)
testDataSparsed = matrix(0, length(testData[,1]),0)


for(i in 1:length(trainData[1,])){
	print(i)
	if(is.numeric(trainData[1,i]) == FALSE){

		uv = unique(trainData[,i])
		write(c(colnames(trainData[i]),i),"dataDescription.txt",append=TRUE,ncolumns=2,sep=":")
		write(uv,"dataDescription.txt",append=TRUE, ncolumns=length(uv),sep="|")
		write("\n--------\n","dataDescription.txt",append=TRUE)

		for(j in 1:length(uv)){
			indicies = which(uv[j]==trainData[,i])
			for(ind in indicies){
				trainData[ind,i] = j
			}
		}
		for(j in 1:length(uv)){
			indicies = which(uv[j]==testData[,i])
			for(ind in indicies){
				testData[ind,i] = j
			}
		}

		tempMatrixTrain = matrix(0,length(trainData[,i]),length(uv))
		colnames(tempMatrixTrain)<-as.list(uv)
		tempMatrixTest = matrix(0,length(testData[,i]),length(uv))
		colnames(tempMatrixTest)<-as.list(uv)
		for(j in 1:length(trainData[,i])){
			tempMatrixTrain[j,as.integer(trainData[j,i])] = 1
		}
		for(j in 1:length(testData[,i])){
			tempMatrixTest[j,as.integer(testData[j,i])] = 1
		}
		trainDataSparsed = cbind(trainDataSparsed,tempMatrixTrain)
		testDataSparsed = cbind(testDataSparsed,tempMatrixTest)

		# for(j in 1:length(trainData[,i])){
		# 	if(is.na(trainData[j,i])){
		# 		trainData[j,i] = 0
		# 	}
		# }
		# for(j in 1:length(testData[,i])){
		# 	if(is.na(testData[j,i])){
		# 		testData[j,i] = 0
		# 	}
		# }
	} else {
		trainDataSparsed = cbind(trainDataSparsed,trainData[,i])
		colnames(trainDataSparsed)[length(trainDataSparsed[1,])] <- colnames(trainData)[i]
		if(i < 81){
			testDataSparsed = cbind(testDataSparsed,testData[,i])
			colnames(testDataSparsed)[length(testDataSparsed[1,])] <- colnames(testData)[i]
		}
	}
}

write.csv(trainDataSparsed, "trainCleanSparse.csv", quote=FALSE, row.names = F)
write.csv(testDataSparsed, "testCleanSparse.csv", quote=FALSE, row.names = F)
