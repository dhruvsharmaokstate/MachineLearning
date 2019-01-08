#Read installed packages
library(partykit)
library(kknn)
library(Hmisc)
library(libcoin)
library(pre)
library(arm)
library(stats)
library(rpart)
library(dplyr)
library(pastecs)
library(skimr)
library(MLmetrics)
library(ISLR)
library(tree)
library(olsrr)
library(tidyr)
library(BBmisc)
library(mice)
library(VIM)
library(rpart.plot)
library(caret)
library(glmnet)
library(e1071)

#Set current working directory
setwd("D:/Misc/Kaggle/Kyphosis")

#Read Kyphosis file as csv
kyphosis_train <- read.csv("kyphosis.csv")

#View the dataframes to see the structure 
View(kyphosis_train)

#Describe function provides descriptive statistics of data
df <- describe(kyphosis_train, exclude.missing = TRUE, digits = 4)
summary(df)

#Viewing basic summary statistics of kyphosis data 
stat.desc(kyphosis_train)
skim(kyphosis_train)

#Plot to see if target variable is normal
hist(kyphosis_train$Kyphosis)

#Plot to see barplots of various fields against kyphosis
counts <- table(kyphosis_train$Age, kyphosis_train$Kyphosis)
barplot(counts, beside = TRUE, col = c("turquoise3","wheat4","slategray"))

counts <- table(kyphosis_train$Number, kyphosis_train$Kyphosis)
barplot(counts, beside = TRUE, col = c("turquoise3","wheat4","slategray"))

counts <- table(kyphosis_train$Start, kyphosis_train$Kyphosis)
barplot(counts, beside = TRUE, col = c("turquoise3","wheat4","slategray"))

#Plot to see Age distribution
d <- density (kyphosis_train$Age, na.rm = TRUE)
plot(d, main = "Age density", xlab = "Age")
polygon(d,col="lightblue",border="red")

#Plot to see Number distribution
d <- density (kyphosis_train$Number, na.rm = TRUE)
plot(d, main = "Number density", xlab = "Number")
polygon(d,col="lightblue",border="red")

#Plot to see Start distribution
d <- density (kyphosis_train$Start , na.rm = TRUE)
plot(d, main = "Number density", xlab = "Start")
polygon(d,col="lightblue",border="red")

#plot a decision tree to see the distribution and variable importance
model_f   <- train( Kyphosis ~ Number + Age + Start,  data = kyphosis_train, method="rpart", na.action = na.pass, tuneLength = 5);
rpart.plot(model_f$finalModel)

#check to see estimates derived from the model
print(model_f$results)

#Boxplot to see Age distribution for various classes
boxplot(kyphosis_train$Age ~ kyphosis_train$Kyphosis)

#Corrplot to see correlation plot from a data matrix
corrplot(kyphosis_train,varnames = NULL, cutpts = NULL,abs = TRUE, details = TRUE, n.col.legend = 5, cex.col = 0.7, cex.var = 0.9, digits = 1, color = FALSE)
require(lattice)
require(ggplot2)
pairs(kyphosis_train, pch = 21)

#Check correlation table and analyze which variables are highly correlated
cor(kyphosis_train)

set.seed(123)
samp_size <- floor(0.70*nrow(kyphosis_train))
train_ind <- sample(seq_len(nrow(kyphosis_train)), size = samp_size)
train <- kyphosis_train[train_ind,] 
validate <- kyphosis_train[-train_ind,]

#Implementing a decision tree 
set.seed(123)
dtree <- rpart(Kyphosis~.,data = train, method = "class")
rpart.plot(dtree, extra = 3, fallen.leaves = TRUE)

dtree_pred <- predict(dtree, data = validate, type = "class")
confusionMatrix(factor(dtree_pred), factor(train$Kyphosis))

#Implementing random forest model
kyphosis.model <- randomForest::randomForest(Kyphosis~Age + Number + Start, data = kyphosis)
print(kyphosis.model)
importance(kyphosis.model)
kyphosis.new <- predict(kyphosis.model, newdata = validate)
print(kyphosis.new)

#Regression Tree
fitK <- rpart(Kyphosis~Age + Number + Start, method = 'anova', data = kyphosis_train)
printcp(fitK)
plotcp(fitK)
summary(fitK)

#Plot tree
plot(fitK,uniform = TRUE, main = "Regression Tree for Kyphosis")
text(fitK, use.n = TRUE, all = TRUE, cex = 0.8)

#Prune the tree
pfitk <- prune(fitK, cp = 0.04940363)
plot(pfitk, uniform = TRUE, mail = 'Prune Regression Tree for Kyphosis')
text(pfitk,use.n = TRUE, all = TRUE, cex = 0.8)
post(pfitk, file = 'ktree2.ps', title = 'Pruned Regression Tree for Kyphosis')

#Conditional Inference Tree for Kyphosis
fit2k <- ctree(Kyphosis~Age + Number + Start, data = kyphosis)

#Weighted KKNN
kyphosis.kknn <- kknn(Kyphosis~.,train = train, test = validate, distance = 1, kernel = 'triangular')
head(kyphosis.kknn$fitted.values)
fitkyph <- fitted(kyphosis.kknn)
table(validate$Kyphosis,fitkyph)
pcol <- as.character(as.numeric(validate$Kyphosis))
pairs(validate,pch = pcol, col = c('green','red'))