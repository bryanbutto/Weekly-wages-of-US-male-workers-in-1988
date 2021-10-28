
install.packages("faraway")

library("faraway")

uswages
?uswages

head(uswages)

str(uswages)

summary(uswages)

install.packages("pastecs")

library(pastecs)

round(stat.desc(uswages),2)

pairs(uswages)

round(cor(uswages),2)

library(corrplot)
cors <- cor(uswages)
corrplot(cors)
corrplot(cors, method = "number")

library(caret)

featurePlot(x=uswages[,-1], y=uswages$wage, type = c("g", "smooth"))
featurePlot(x=uswages[,-1], y=uswages$wage, type = c("g", "p", "smooth"))

library(ggplot2)

ggplot(uswages, aes(x=wage)) + geom_histogram(alpha=2) + ggtitle("wage")
ggplot(uswages, aes(x=wage)) + geom_density() + ggtitle("wage")

ggplot(uswages, aes(wage)) + geom_boxplot() + ggtitle("Wage")
ggplot(uswages, aes(educ)) + geom_boxplot() + ggtitle("Years of Education")
ggplot(uswages, aes(exper)) + geom_boxplot() + ggtitle("Years of Experience")

ggplot(uswages, aes(x=educ, y=wage)) + geom_point() + ggtitle("Wage vs. Years of Education")
ggplot(uswages, aes(x=exper, y=wage)) + geom_point() + ggtitle("Wage vs. Years of Experience")
ggplot(uswages, aes(x=race, y=wage)) + geom_point() + ggtitle("Wage vs. Race")
ggplot(uswages, aes(x=smsa, y=wage)) + geom_point() + ggtitle("Wage vs. Standard Metropolitan Statistcal Area")
ggplot(uswages, aes(x=ne, y=wage)) + geom_point() + ggtitle("Wage vs. Northeast")
ggplot(uswages, aes(x=mw, y=wage)) + geom_point() + ggtitle("Wage vs. Midwest")
ggplot(uswages, aes(x=we, y=wage)) + geom_point() + ggtitle("Wage vs. West")
ggplot(uswages, aes(x=so, y=wage)) + geom_point() + ggtitle("Wage vs. South")
ggplot(uswages, aes(x=pt, y=wage)) + geom_point() + ggtitle("Wage vs. Part time")

uswages$race1 <- ifelse(uswages$race == 1, "Black", "White")
uswages$smsa1 <- ifelse(uswages$smsa == 1, "Yes", "No")
uswages$ne1 <- ifelse(uswages$ne == 1, "Yes", "No")
uswages$mw1 <- ifelse(uswages$mw == 1, "Yes", "No")
uswages$we1 <- ifelse(uswages$we == 1, "Yes", "No")
uswages$so1 <- ifelse(uswages$so == 1, "Yes", "No")
uswages$pt1 <- ifelse(uswages$pt == 1, "Yes", "No")

head(uswages, 5)

str(uswages)

ggplot(uswages, aes(x=wage, color=race1)) + geom_density() + ggtitle("Wage by Race")
ggplot(uswages, aes(x=wage, color=smsa1)) + geom_density() + ggtitle("Wage by Standard Metropolitan Statistical Area")
ggplot(uswages, aes(x=wage, color=ne1)) + geom_density() + ggtitle("Wage by Northeast")
ggplot(uswages, aes(x=wage, color=mw1)) + geom_density() + ggtitle("Wage by Midwest")
ggplot(uswages, aes(x=wage, color=we1)) + geom_density() + ggtitle("Wage by West")
ggplot(uswages, aes(x=wage, color=so1)) + geom_density() + ggtitle("Wage by South")
ggplot(uswages, aes(x=wage, color=pt1)) + geom_density() + ggtitle("Wage by Part time")

install.packages("nortest")

library(nortest)

pearson.test(uswages$wage)

mean(uswages$wage)+3*sd(uswages$wage)

wage.outlier <- uswages[uswages$wage > 1987.61, ]
show(wage.outlier)

str(uswages)

uswages.omit <- na.omit(uswages)
str(uswages.omit)

wage2 <- log(uswages$wage)
head(wage2)

ggplot(uswages, aes(x=wage2)) + geom_histogram(alpha=2) + ggtitle("Wage log")
ggplot(uswages, aes(x=wage2)) + geom_density() + ggtitle("Wage log")

pearson.test(wage2)

mean(wage2)+3*sd(wage2)

wage2.outlier <- uswages[wage2 > 8.35, ]
show(wage2.outlier)

install.packages("caret")

library(caret)

set.seed(602)
uswages.train <- createDataPartition(uswages$wage, p=3/4, list = FALSE)
head(uswages.train, 10)
tail(uswages.train, 10)

trainingset <- uswages[uswages.train,]
testingset <- uswages[-uswages.train, ]
str(trainingset)
str(testingset)
head(trainingset)
head(testingset)

repeatedsplits <- createDataPartition(trainingset$wage,p=.8, times = 10)
str(repeatedsplits)

repeatedsplits1 <- createFolds(trainingset$wage, k=10, returnTrain=TRUE)
str(repeatedsplits1)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainlm <- train(wage ~ ., data=trainingset, method="lm", trControl = controlobject)
str(modtrainlm)

set.seed(12)
uswages$RN <- rnorm(2000)
uswages$RN10 <- rnorm(2000)*10
fit <- lm(wage ~., data = uswages)
summary(fit)

install.packages("leaps")

library(leaps)

regfitfull <- regsubsets(wage ~ . , data=uswages)
regsummary <- summary(regfitfull)
regsummary

names(regsummary)

round(regsummary$rsq, 2)
round(regsummary$adjr2, 2)
round(regsummary$cp, 2)
round(regsummary$bic, 2)

step(fit)

step(fit, direction="backward")

modols <- lm(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data = trainingset)
summary(modols)

names(modols)

show(rmsemodols <- RMSE(trainingset$wage, modols$fitted.values))

show(maemodols <- MAE(trainingset$wage, modols$fitted.values))

show(r2modols <- (cor(trainingset$wage, modols$fitted.values))^2)

varImp(modols)

modolsp <- predict(modols, trainingset)
head(modolsp)

modolspm <- data.frame(obs=trainingset$wage, pred=modolsp)
head(modolspm)

defaultSummary(modolspm)

plot(modolspm$pred, modolspm$obs, abline(a=0, b=1, col="red"))

modolspm.resid <- (modolspm$obs - modolspm$pred)
plot(modolspm$pred, modolspm.resid, abline(h=0, col="red"))

modolsp <- predict(modols, testingset)
head(modolsp)

modolspm <- data.frame(obs=testingset$wage, pred=modolsp)
head(modolspm)

defaultSummary(modolspm)

plot(modolspm$pred, modolspm$obs, abline(a=0, b=1, col="red"))

modolspm.resid <- (modolspm$obs - modolspm$pred)
plot(modolspm$pred, modolspm.resid, abline(h=0, col="red"))

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainols <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="lm",
                    trControl = controlobject)
summary(modtrainols)

modtrainols

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainrlm <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="rlm",
                    preProc = c("center", "scale"), trControl = controlobject)
summary(modtrainrlm)

modtrainrlm

trainingsetx <- trainingset[,2:10]
pctrainingsetx <- prcomp(trainingsetx)
summary(pctrainingsetx)

names(pctrainingsetx)

round(pctrainingsetx$rotation,2)

round(pctrainingsetx$rot,2)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainrlmpc <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="lm",
                    preProc = "pca", trControl = controlobject)
summary(modtrainrlmpc)

modtrainrlmpc

varImp(modtrainrlmpc)

names(modtrainrlmpc)

modtrainrlmpc$preProcess

names(testingset)

modtrainrlmpcp <- predict(modtrainrlmpc, testingset)
modtrainrlmpcpm <- data.frame(obs = testingset$wage, pred = modtrainrlmpcp)
modtrainrlmpcpm.sum <- defaultSummary(modtrainrlmpcpm)
modtrainrlmpcpm.sum

install.packages("pls")

library(pls)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainrlmpls <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="pls",
                    trControl = controlobject)
summary(modtrainrlmpls)

modtrainrlmpls

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainrlmpls <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="pls",
                    tuneLength=15, trControl = controlobject)
summary(modtrainrlmpls)

modtrainrlmpls

plot(modtrainrlmpls)

varImp(modtrainrlmpls)

install.packages("elasticnet")

library(elasticnet)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainrlmrr <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="ridge",
                    trControl = controlobject)
summary(modtrainrlmrr)

modtrainrlmrr

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainrlmlasso <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="lasso",
                    trControl = controlobject)
summary(modtrainrlmlasso)

modtrainrlmlasso

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainrlmenet <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="enet",
                    trControl = controlobject)
summary(modtrainrlmenet)

modtrainrlmenet

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainnn <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="avNNet",
                    linout=TRUE, trace=FALSE, trControl = controlobject)
summary(modtrainnn)

modtrainnn

plot(modtrainnn)

varImp(modtrainnn)
plot(varImp(modtrainnn))

install.packages("plotmo")

library(plotmo)

plotmo(modtrainnn)

plotres(modtrainnn)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
nnetgrid <- expand.grid(.decay = c(.001, .01, .1), .size = seq(1, 27, by = 2), .bag = "FALSE")
set.seed(444)
modtrainnn2 <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="avNNet",
                    linout=TRUE, trace=FALSE, maxit = 300, tuneGrid = nnetgrid, trControl = controlobject)
summary(modtrainnn2)

modtrainnn2

plot(modtrainnn2)

varImp(modtrainnn2)
plot(varImp(modtrainnn2))

plotmo(modtrainnn2)

plotres(modtrainnn2)

install.packages("earth")

library(earth)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainmars <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="earth",
                    trControl = controlobject)
summary(modtrainmars)

modtrainmars

plot(modtrainmars)

varImp(modtrainmars)
plot(varImp(modtrainmars))

plotmo(modtrainmars)

plotres(modtrainmars)

install.packages("kernlab")

library(kernlab)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainsvm <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="svmRadial",
                    trControl = controlobject)
summary(modtrainsvm)

modtrainsvm

plot(modtrainsvm)

varImp(modtrainsvm)
plot(varImp(modtrainsvm))

plotmo(modtrainsvm)

plotres(modtrainsvm)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainsvm2 <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="svmRadial",
                    trControl = controlobject, tuneLength = 15)
summary(modtrainsvm2)

modtrainsvm2

plot(modtrainsvm2)

varImp(modtrainsvm2)
plot(varImp(modtrainsvm2))

plotmo(modtrainsvm2)

plotres(modtrainsvm2)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainknn <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="knn",
                    trControl = controlobject)
summary(modtrainknn)

modtrainknn

plot(modtrainknn)

varImp(modtrainknn)
plot(varImp(modtrainknn))

plotmo(modtrainknn)

plotres(modtrainknn)

install.packages("rpart")

library(rpart)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtraincart <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="rpart",
                    trControl = controlobject)
summary(modtraincart)

modtraincart

install.packages("rpart.plot")

library(rpart.plot)

plot(modtraincart)

varImp(modtraincart)
plot(varImp(modtraincart))

plotmo(modtraincart)

plotres(modtraincart)

install.packages("party")

library(party)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainctree <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="ctree",
                    trControl = controlobject)
summary(modtrainctree)

modtrainctree

plot(modtrainctree)

varImp(modtrainctree)
plot(varImp(modtrainctree))

plotmo(modtrainctree)

plotres(modtrainctree)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtraintreebag <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="treebag",
                    trControl = controlobject)
summary(modtraintreebag)

modtraintreebag

varImp(modtraintreebag)
plot(varImp(modtraintreebag))

plotmo(modtraintreebag)

plotres(modtraintreebag)

install.packages("randomForest")

library(randomForest)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainrf <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="rf",
                    trControl = controlobject)
summary(modtrainrf)

modtrainrf

plot(modtrainrf)

varImp(modtrainrf)
plot(varImp(modtrainrf))

plotmo(modtrainrf)

plotres(modtrainrf)

install.packages("gbm")

library(gbm)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtrainboost <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="gbm",
                    verbose=FALSE, trControl = controlobject)
summary(modtrainboost)

modtrainboost

plot(modtrainboost)

varImp(modtrainboost)
plot(varImp(modtrainboost))

plotmo(modtrainboost)

plotres(modtrainboost)

install.packages("Cubist")

library(Cubist)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(444)
modtraincubist <- train(wage ~ educ + exper + race + smsa + ne + mw + so + pt, data=trainingset, method="cubist",
                    verbose=FALSE, trControl = controlobject)
summary(modtraincubist)

modtraincubist

plot(modtraincubist)

varImp(modtraincubist)
plot(varImp(modtraincubist))

plotmo(modtraincubist)

plotres(modtraincubist)

installed.packages()

library(caret)

ressum <- resamples(list("LMOLS" = modtrainols, 
                         "LMRobust" = modtrainrlm,
                         "LMPCA" = modtrainrlmpc,
                         "LMPLS" = modtrainrlmpls,
                         "LMRidge" = modtrainrlmrr,
                         "LMLasso" = modtrainrlmlasso,
                         "LMENet" = modtrainrlmenet,
                         "NLNN" = modtrainnn,
                         "NLMars" = modtrainmars,
                         "NLSVM" = modtrainsvm,
                         "NLKNN" = modtrainknn,
                         "TrCART" = modtraincart,
                         "TrCTree" = modtrainctree,
                         "TrBag" = modtraintreebag,
                         "TrRF" = modtrainrf,
                         "TrBoost" = modtrainboost,
                         "TrCubist" = modtraincubist))

names(ressum)

ressum$metrics

ressum$values

parallelplot(ressum, metric="MAE")
parallelplot(ressum, metric="RMSE")
parallelplot(ressum, metric="Rsquared")

ressum1 <- summary(ressum)

names(ressum1)

names(ressum1$statistics)

ressum.rmse <- round(ressum1$statistics$RMSE, 2)
ressum.rmse

ressum.rmse <- ressum.rmse[ , 4]

ressum.rmse <- round(ressum.rmse, 2)
ressum.rmse

ressum.r2 <- round(ressum1$statistics$Rsquared, 2)
ressum.r2

ressum.r2 <- ressum.r2[ , 4]

ressum.r2 <- round(ressum.r2, 2)
ressum.r2

ressum.train <- cbind(ressum.rmse, ressum.r2)
ressum.train

## LMOLS
modtrainols.pred <- predict(modtrainols, testingset)
modtrainols.test <- data.frame(obs = testingset$wage, pred = modtrainols.pred)
modtrainols.stats <- defaultSummary(modtrainols.test)
round(modtrainols.stats, 2)

## LMRobust
modtrainrlm.pred <- predict(modtrainrlm, testingset)
modtrainrlm.test <- data.frame(obs = testingset$wage, pred = modtrainrlm.pred)
modtrainrlm.stats <- defaultSummary(modtrainrlm.test)
round(modtrainrlm.stats, 2)

## LMPCA
modtrainrlmpc.pred <- predict(modtrainrlmpc, testingset)
modtrainrlmpc.test <- data.frame(obs = testingset$wage, pred = modtrainrlmpc.pred)
modtrainrlmpc.stats <- defaultSummary(modtrainrlmpc.test)
round(modtrainrlmpc.stats, 2)

## LMPLS
modtrainrlmpls.pred <- predict(modtrainrlmpls, testingset)
modtrainrlmpls.test <- data.frame(obs = testingset$wage, pred = modtrainrlmpls.pred)
modtrainrlmpls.stats <- defaultSummary(modtrainrlmpls.test)
round(modtrainrlmpls.stats, 2)

## LMRidge
modtrainrlmrr.pred <- predict(modtrainrlmrr, testingset)
modtrainrlmrr.test <- data.frame(obs = testingset$wage, pred = modtrainrlmrr.pred)
modtrainrlmrr.stats <- defaultSummary(modtrainrlmrr.test)
round(modtrainrlmrr.stats, 2)

## LMLasso
modtrainrlmlasso.pred <- predict(modtrainrlmlasso, testingset)
modtrainrlmlasso.test <- data.frame(obs = testingset$wage, pred = modtrainrlmlasso.pred)
modtrainrlmlasso.stats <- defaultSummary(modtrainrlmlasso.test)
round(modtrainrlmlasso.stats, 2)

## LMENet
modtrainrlmenet.pred <- predict(modtrainrlmenet, testingset)
modtrainrlmenet.test <- data.frame(obs = testingset$wage, pred = modtrainrlmenet.pred)
modtrainrlmenet.stats <- defaultSummary(modtrainrlmenet.test)
round(modtrainrlmenet.stats, 2)

## NLNN
modtrainnn.pred <- predict(modtrainnn, testingset)
modtrainnn.test <- data.frame(obs = testingset$wage, pred = modtrainnn.pred)
modtrainnn.stats <- defaultSummary(modtrainnn.test)
round(modtrainnn.stats, 2)

## NLMars
modtrainmars.pred <- predict(modtrainmars, testingset)
modtrainmars.test <- data.frame(obs = testingset$wage, pred = modtrainmars.pred)
modtrainmars.test$pred <- modtrainmars.test$y
modtrainmars.stats <- defaultSummary(modtrainmars.test)
round(modtrainmars.stats, 2)

## NLSVM
modtrainsvm.pred <- predict(modtrainsvm, testingset)
modtrainsvm.test <- data.frame(obs = testingset$wage, pred = modtrainsvm.pred)
modtrainsvm.stats <- defaultSummary(modtrainsvm.test)
round(modtrainsvm.stats, 2)

## NLKNN
modtrainknn.pred <- predict(modtrainknn, testingset)
modtrainknn.test <- data.frame(obs = testingset$wage, pred = modtrainknn.pred)
modtrainknn.stats <- defaultSummary(modtrainknn.test)
round(modtrainknn.stats, 2)

## TrCART
modtraincart.pred <- predict(modtraincart, testingset)
modtraincart.test <- data.frame(obs = testingset$wage, pred = modtraincart.pred)
modtraincart.stats <- defaultSummary(modtraincart.test)
round(modtraincart.stats, 2)

## TrCTree
modtrainctree.pred <- predict(modtrainctree, testingset)
modtrainctree.test <- data.frame(obs = testingset$wage, pred = modtrainctree.pred)
modtrainctree.stats <- defaultSummary(modtrainctree.test)
round(modtrainctree.stats, 2)

## TrBag
modtraintreebag.pred <- predict(modtraintreebag, testingset)
modtraintreebag.test <- data.frame(obs = testingset$wage, pred = modtraintreebag.pred)
modtraintreebag.stats <- defaultSummary(modtraintreebag.test)
round(modtraintreebag.stats, 2)

## TrRF
modtrainrf.pred <- predict(modtrainrf, testingset)
modtrainrf.test <- data.frame(obs = testingset$wage, pred = modtrainrf.pred)
modtrainrf.stats <- defaultSummary(modtrainrf.test)
round(modtrainrf.stats, 2)

## TrBoost
modtrainboost.pred <- predict(modtrainboost, testingset)
modtrainboost.test <- data.frame(obs = testingset$wage, pred = modtrainboost.pred)
modtrainboost.stats <- defaultSummary(modtrainboost.test)
round(modtrainboost.stats, 2)

## TrCubist
modtraincubist.pred <- predict(modtraincubist, testingset)
modtraincubist.test <- data.frame(obs = testingset$wage, pred = modtraincubist.pred)
modtraincubist.stats <- defaultSummary(modtraincubist.test)
round(modtraincubist.stats, 2)

ressum.test <- data.frame(modtrainols.stats, 
                         modtrainrlm.stats,
                         modtrainrlmpc.stats,
                         modtrainrlmpls.stats,
                         modtrainrlmrr.stats,
                         modtrainrlmlasso.stats,
                         modtrainrlmenet.stats,
                         modtrainnn.stats,
                         modtrainmars.stats,
                         modtrainsvm.stats,
                         modtrainknn.stats,
                         modtraincart.stats,
                         modtrainctree.stats,
                         modtraintreebag.stats,
                         modtrainrf.stats,
                         modtrainboost.stats,
                         modtraincubist.stats)
ressum.test

ressum.test1 <- t(ressum.test)
round(ressum.test1, 2)

str(ressum.test1)

ressum.test1 <- data.frame(ressum.test1)
str(ressum.test1)

ressum.test1$RMSE <- round(ressum.test1$RMSE, 2)
ressum.test1$Rsquared <- round(ressum.test1$Rsquared, 2)
ressum.test1$MAE <- round(ressum.test1$MAE, 2)

ressum.test1$Model <- row.names(ressum.test1)
row.names(ressum.test1) <- seq(1:17)
ressum.test1$Model <- gsub(".stats", "", ressum.test1$Model)
ressum.test1$Model <- gsub("modtrain", "", ressum.test1$Model)
ressum.test1$Model <- format(ressum.test1$Model, justify="left")
ressum.test1[ , c(4, 1:3)]

ressum.test2 <- ressum.test1[order(ressum.test1$RMSE, decreasing = FALSE), ]
ressum.test2 [ , c(4,1:3)]

ggplot(ressum.test1,aes(x= reorder(Model,-RMSE), y=RMSE))+
geom_bar(stat ="identity") + coord_flip()

ggplot(ressum.test1,aes(x= reorder(Model,Rsquared), y=Rsquared)) +
geom_bar(stat ="identity") + coord_flip()
