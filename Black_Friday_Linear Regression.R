#Install necessary packages
install.packages("Hmisc")
install.packages("pre")
install.packages("libcoin")
install.packages("arm")
install.packages("stats")
install.packages("rpart")
install.packages("dplyr")
install.packages("Rcpp")
install.packages("pastecs")
install.packages("skimr")
install.packages("MLmetrics")
install.packages("ISLR")
install.packages("tree")
install.packages("olsrr")
install.packages("BBmisc")

#Read installed packages 
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

# Gen and print current working directory
print(getwd())

#Set current working directory
setwd("D:/Misc/Kaggle/Black Friday")

#Get and Print Current working directory
print(getwd())

#Read Sales file as csv
bf_train <- read.csv("train.csv")
bf_test <- read.csv("test.csv")

#View the dataframes to see the structure 
View(bf_train)
View(bf_test)

#Describe function provides descriptive statistics of data
df <- describe(bf_train, exclude.missing = TRUE, digits = 4)
stat.desc(bf_train)
skim(bf_train)

#Plot to see if target variable is normal
hist(bf_train$Purchase)
bf_train$Purchase <- normalize(bf_train$Purchase, method = "standardize", range = c(0,1), margin = 1L, on.constant = "quiet")
bf_train$Age <- normalize(bf_train$Age, method = "standardize", range = c(0,1), margin = 1L, on.constant = "quiet")
plot(bf_train$Age)

#Drop UserID and Product_ID 
bf_train$User_ID <- NULL
bf_train$Product_ID <- NULL

#Check correlation table and analyze which variables are highly correlated
cor(bf_train)

#Principle Component Analysis but first make NA as 0 from Product Categories
bf_train[is.na(bf_train)] <- 0
pcaData <- princomp(bf_train, scores = TRUE, cor = TRUE)
summary(pcaData)
loadings(pcaData)
screeplot(pcaData, type= "line", main = "Screeplot")

#Plot to see effect of Age on Purchase Amount
plot(bf_train$Age, bf_train$Purchase)

#Plot to see effect of Gender on Purchase Amount
plot(bf_train$Gender, bf_train$Purchase)

#Plot to see effect of Years in Current city on Purchase Amount
plot(bf_train$Stay_In_Current_City_Years, bf_train$Purchase)

#Plot to see effect of City Category on Purchase Amount
plot(bf_train$City_Category, bf_train$Purchase)

#Recode gender female and male into 0 and 1 and convert factor into integer

bf_train$Gender <- factor(bf_train$Gender, levels = c("F","M"), labels=c(0,1))
bf_train$Gender <- as.factor(bf_train$Gender)
bf_test$Gender <- factor(bf_test$Gender, levels = c("F","M"), labels=c(0,1))
bf_test$Gender <- as.integer(bf_test$Gender)

#Recode Age groups into numeric and convert factor into integer
bf_train$Age <- factor(bf_train$Age, levels = c("0-17","18-25","26-35","36-45","46-50","51-55","55+"), labels=c(0,1,2,3,4,5,6))
bf_train$Age <- as.factor(bf_train$Age)

bf_test$Age <- factor(bf_test$Age, levels = c("0-17","18-25","26-35","36-45","46-50","51-55","55+"), labels=c(0,1,2,3,4,5,6))
bf_test$Age <- as.integer(bf_test$Age)

#Recode City Category into numeric and convert factor into integer
bf_train$City_Category <- factor(bf_train$City_Category, levels = c("A","B","C"), labels=c(0,1,2))
bf_train$City_Category <- as.factor(bf_train$City_Category)

bf_test$City_Category <- factor(bf_test$City_Category, levels = c("A","B","C"), labels=c(0,1,2))
bf_test$City_Category <- as.factor(bf_test$City_Category)



#Recode Stay in current city years into numeric and convert factor into integer
bf_train$Stay_In_Current_City_Years <- factor(bf_train$Stay_In_Current_City_Years, levels = c("0","1","2","3","4+"), labels=c(0,1,2,3,4))
bf_train$Stay_In_Current_City_Years <- as.factor(bf_train$Stay_In_Current_City_Years)

bf_train$Marital_Status <- as.factor(bf_train$Marital_Status)


bf_test$Stay_In_Current_City_Years <- factor(bf_test$Stay_In_Current_City_Years, levels = c("0","1","2","3","4+"), labels=c(0,1,2,3,4))
bf_test$Stay_In_Current_City_Years <- as.integer(bf_test$Stay_In_Current_City_Years)

#Dividing independent variables and target variables
x = bf_train[c("Gender","Age","Occupation","City_Category","Stay_In_Current_City_Years","Marital_Status","Purchase","Product_Category_1","Product_Category_2","Product_Category_3")]
y = bf_train["Purchase"]

#To observe correlation between different variables
bf_train1[is.na(bf_train1)] <- 0
cor(bf_train1,use = "complete.obs",method="kendall")

#Divide dataset into training and validation
bf_sample <- sample(1:nrow(bf_train), size = floor(nrow(bf_train)*0.7))
bftrain <- bf_train[bf_sample,]
bftest <- bf_train[-bf_sample,]


#Creating model using train set
lmbftrain <- lm(Purchase~.,data = bftrain)
lmbftrain

#Decision Tree to see important variables
tree.bftrain <- tree(bftrain$Purchase~., data = bftrain)
summary(tree.bftrain)
plot(tree.bftrain)
text(tree.bftrain, pretty = 0)
tree.bftrain

#Model to select important variables 
model <- lm(bftrain$Purchase ~ bftrain$Gender + bftrain$Age + bftrain$Occupation + bftrain$City_Category + bftrain$Stay_In_Current_City_Years + bftrain$Marital_Status + bftrain$Product_Category_1, data=bftrain)
summary(model)
k<- ols_step_all_possible(model)
k
plot(k)
l <- ols_step_best_subset(model)
l

#prediction <- predict(model, newdata = bftest)
#head(prediction)
#SSE <- sum((bftest$Purchase - prediction) ^ 2)
#SST <- sum((bftest$Purchase - mean(bftest$Purchase)) ^ 2)
#1-SSE/SST

train_prob_purchase = predict(model,newdata = bftrain)
bftrain = cbind(bftrain,train_prob_purchase)
bftrain

#MAPE
(mean(abs((bftrain$Purchase-bftrain$train_prob_purchase)/bftrain$Purchase)))

#RMSE
(sqrt(mean((bftrain$Purchase - bftrain$train_prob_purchase)**2)))

#ANOVA for various variables
anova(model)

#Plot for residuals

