

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
library(mice)
library(VIM)
library(rpart.plot)
library(caret)
library(glmnet)
library(e1071)

# Gen and print current working directory
print(getwd())

#Set current working directory
setwd("D:/Misc/Kaggle/Titanic")

#Get and Print Current working directory
print(getwd())

#Read Titanic Survival file as csv
titanic_train <- read.csv("train.csv")
titanic_test <- read.csv("test.csv")

#View the dataframes to see the structure 
View(titanic_train)

#Describe function provides descriptive statistics of data
df <- describe(titanic_train, exclude.missing = TRUE, digits = 4)
summary(df)
stat.desc(titanic_train)
skim(titanic_train)

#Plot to see if target variable is normal
hist(titanic_train$Survived)
#titanic_train$Survived <- normalize(titanic_train$Survived, method = "standardize", range = c(0,1), margin = 1L, on.constant = "quiet")

#Plot to see what age group of people fared better at survival
counts <- table(titanic_train$Age, titanic_train$Survived)
barplot(counts, beside = TRUE, col = c("turquoise3","wheat4","slategray"))

#Plot to see Age distribution
d <- density (titanic_train$Age, na.rm = TRUE)
plot(d, main = "Age density", xlab = "Age")
polygon(d,col="lightblue",border="red")

#Plot to see Fare distribution
d <- density (titanic_train$Fare, na.rm = TRUE)
plot(d, main = "Fare density", xlab = "Fare")
polygon(d,col="lightblue",border="red")

#Treat Missing values in Age using MICE package
titanic.mis <- subset(titanic_train, select = -c(PassengerId))
summary(titanic.mis)
md.pattern(titanic.mis)

#Create a new dataset age 

age <- titanic_train$Age
n = length(age)
# replace missing value with a random sample from raw data
set.seed(123)
for(i in 1:n){
  if(is.na(age[i])){
    age[i] = sample(na.omit(titanic_train$Age),1)
  }
}
titanic_train$Age <- age

#Check if there are missing values in any other field
which(is.na(titanic_train))

#plot a decision tree to see the distribution and variable importance
model_f   <- train( Fare ~ Pclass + Sex + Embarked + SibSp + Parch, data = titanic_train, method="rpart", na.action = na.pass, tuneLength = 5);
rpart.plot(model_f$finalModel)

#check to see estimates derived from the model
print(model_f$results)

#Create a new category called minor which indicates children
child <- 14
titanic_train$Minor <- ifelse(titanic_train$Age < child, 1, 0)
titanic_train$Minor <- ifelse(is.na(titanic_train$Minor),0,titanic_train$Minor)

#Convert Sex from factor to integer
titanic_train$Sex <- as.integer(ifelse(titanic_train$Sex == 'male', 1, 0))
titanic_test$Sex <- as.integer(ifelse(titanic_test$Sex == 'male', 1, 0))


#Create a column for surnames
titanic_train$Surname <- sapply(as.character(titanic_train$Name), FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]});

#Check to see if any families traveled with RMS Titanic
table(titanic_train$Surname ~ titanic_train$Pclass)


#Boxplot to see Age distribution for various classes
boxplot(titanic_train$Age ~ titanic_train$Pclass)

#Plot to see effect of Class on Survival
counts <- table(titanic_train$Pclass , titanic_train$Survived)
barplot(counts, beside = TRUE, col = c("turquoise3","wheat4","slategray"))

#Plot to see effect of family size on survival
counts <- table(titanic_train$Parch , titanic_train$Survived)
barplot(counts, beside = TRUE, col = c("turquoise3","wheat4","slategray"))

na.omit(titanic_train)
#Drop PassengerId, Name, Ticket, Embarked, Cabin and Surname 
titanic_train$Name <- NULL
titanic_train$Ticket <- NULL
titanic_train$Cabin <- NULL
titanic_train$Surname <- NULL
titanic_train$Embarked <- NULL

titanic_test$Name <- NULL
titanic_test$Ticket <- NULL
titanic_test$Cabin <- NULL
titanic_test$Surname <- NULL
titanic_test$Embarked <- NULL

#Combine Parch and SibSp columns into one to create Family column
titanic_train$Family <- titanic_train$SibSp + titanic_train$Parch 
titanic_test$Family <- titanic_test$SibSp + titanic_test$Parch 


#Now drop SibSp and Parch to avoid multicollinearity
titanic_train$SibSp <- NULL
titanic_train$Parch <- NULL
titanic_test$Parch <- NULL
titanic_test$SibSp <- NULL
#Check correlation table and analyze which variables are highly correlated
cor(titanic_train)

#Make tables to see effects of various variables on Survival
tapply(titanic_train$Survived, titanic_train$Sex, mean)
tapply(titanic_train$Survived, titanic_train$Family, mean)

set.seed(123)
samp_size <- floor(0.70*nrow(titanic_train))
train_ind <- sample(seq_len(nrow(titanic_train)), size = samp_size)
train <- titanic_train[train_ind,] 
validate <- titanic_train[-train_ind,]

#Implementing a random forest 
set.seed(123)
dtree <- rpart(Survived~.,data = train, method = "class")
rpart.plot(dtree, extra = 3, fallen.leaves = TRUE)

dtree_pred <- predict(dtree, data = train, type = "class")
confusionMatrix(factor(dtree_pred), factor(train$Survived))


#Implementing a logistic regression model
lmodel <- glm(Survived~., family = binomial(link = logit), data = train)
summary(lmodel)

#Estimating R2 of the model
train$Survived <- as.integer(train$Survived)
AAA <- predict(lmodel, newdata = validate, type = "response")
View(AAA)
RMSE2 <- mean((train$Survived) - AAA)^2
RMSE2
#Estimating Accuracy of the model
table(train$Survived, AAA > 0.5)
(333 + 170)/(333 + 49 + 71 + 170)

#Lasso Regression
x = data.matrix(train[,-train$Survived])
y = train$Survived

ridge = glmnet(x,y,family = "binomial", alpha = 0)
lasso = glmnet(x,y,family = "binomial", alpha = 1)
summary(ridge)
plot(ridge, main = "Ridge")
BBB <- predict(ridge, newdata = train, newx = x, type = "response")
R2 <- mean((train$Survived) - BBB)/2 
R2

#SVM
set.seed(123)
linear.tune <- tune.svm(Survived~., data = train, ekrnel = "linear",cost=c(0.01,0.1,0.2,0.5,0.7,1,2,3,5,10,15,20,50,100))
best.linear <- linear.tune$best.modeyl
best.test <- predict(best.linear, newdata = train, type = "class")

table(train$Survived,best.test > 0.5)
(326 + 166)/(326+166+56+75)

Survived <- predict(AAA,newdata = validate)

#Implementing a random forest model
titanic.model <- randomForest::randomForest(Survived~Pclass + Sex + Age + Fare + Family, data = train, ntree = 500, mtry = 3, nodesize = 0.01*nrow(train))
Survived <- predict(titanic.model, newdata = titanic_test)

#Submitting predictions to Kaggle
PassengerId <- titanic_test$PassengerId
output.df <- as.data.frame(PassengerId)
output.df$Survived <- Survived
write.csv(output.df,'titanic_kaggle_submissions.csv',row.names = FALSE)
getwd()
