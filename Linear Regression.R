#Install necessary packages
install.packages("Hmisc")
install.packages("pre")
install.packages("libcoin")
install.packages("arm")
install.packages("stats")
install.packages("rpart")

#Read installed packages 
library(Hmisc)
library(libcoin)
library(pre)
library(arm)
library(stats)
library(rpart)

# Gen and print current working directory
print(getwd())

#Set current working directory
setwd("C:/Users/dhruv/Downloads/students-performance-in-exams")

#Get and Print Current working directory
print(getwd())

#Read exam scores file as csv
data <- read.csv("StudentsPerformance.csv")

View(data)
#Summarize function provides descriptive statistics Output table of data
describe(data, exclude.missing = TRUE, digits = 4)

#Feature creation to reate a new variable called overall.score which stores overall scores
data$overall.score <- data$math.score + data$reading.score + data$writing.score

#Create another feature called Average Scores which stores average of all three scores
data$average.score <- (data$math.score + data$reading.score + data$writing.score)/3
#View data to see if new features are created
View(data)

#Too many digits in average score so round them off 
data$average.score <- round(data$average.score,digits = 2)

#Plot to see distribution of parental level of education
plot(School$parental.level.of.education)


#BoxPlot to see effect of gender on average scores
plot(data$gender,data$average.score,notch = TRUE, col=c("gold","darkgreen"),main="Gender vs Average Scores", xlab = "Gender", ylab = "Scores")

#Plot to see effect of race ethnicity on average scores
plot(data$race.ethnicity,data$average.score,notch = TRUE, col=c("gold","darkgreen"),main="Ethnicity vs Average Scores")

#Plot to see effect of parental education on average scores
plot(data$parental.level.of.education,data$average.score,notch = TRUE, col=c("gold","darkgreen"),main="Parental Eduacation vs Average Scores")

#Plot to see effect of lunch on average scores
plot(data$lunch,data$average.score,notch = TRUE, col=c("gold","darkgreen"),main="Lunch vs Average Scores")

#Recode gender female and male into 0 and 1
School <- data
School$gender <- factor(School$gender, levels = c("female","male"), labels=c(0,1))

#Recode race ethnicity groups into numeric
School$race.ethnicity <- factor(School$race.ethnicity, levels = c("group A","group B","group C","group D","group E"), labels=c(0,1,2,3,4))

#Recode parental level of education into numeric
School$parental.level.of.education <- factor(School$parental.level.of.education, levels = c("associate's degree","some high school","high school","some college","bachelor's degree", "master's degree"), labels=c(0,1,2,3,4,5))

#Recode lunch into numeric
School$lunch <- factor(School$lunch, levels = c("free/reduced","standard"), labels=c(0,1))

#Recode test preparation course into numeric
School$test.preparation.course <- factor(School$test.preparation.course, levels = c("completed","none"), labels=c(0,1))

#Derive a prediction rule estimate
pred <- pre(overall.score~gender + race.ethnicity + parental.level.of.education + lunch + test.preparation.course, data, family = "multinomial", use.grad = TRUE, type = "both", sampfrac = 0.5, maxdepth = 3L, learnrate = 0.01, mtry = Inf, ntrees = 500, removeComplements = TRUE)
pred
#To visualize correlation between different variables
corrplot(data, varnames = NULL, cutpts = NULL, abs = TRUE, details = TRUE, n.col.legend = 5, cex.col = 0.7, cex.var = 0.9, color = TRUE)

#Dividing independent variables and target variables
x = School[c("gender","race.ethnicity","parental.level.of.education","lunch","test.preparation.course")]
y = School["average.score"]

#Dividing dataset into train and validation 
x_train <- floor(0.70*nrow(x))
x_validate <- floor(0.30*nrow(x))
y_train <- floor(0.70*nrow(y))
y_validate <- floor(0.30*nrow(y))

#Set seed to make your partition reproducible
set.seed(123)
x_train_ind <- sample(seq_len(nrow(x)),size = x_train)
y_train_ind <- sample(seq_len(nrow(y)),size = y_train)

x_training <- x[x_train_ind,]
x_validation <- x[-x_train_ind,]
y_training <- y[y_train_ind,]
y_validation <- y[-y_train_ind,]

#Creating model using train set
lmschool <- lm(y$average.score~x$gender + x$race.ethnicity + x$parental.level.of.education + x$lunch + x$test.preparation.course)

#ANOVA for various variables
anova(lmschool)

#Plot for residuals
plot(lmschool, las = 1)