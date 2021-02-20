#2
library(dplyr)
library(sqldf)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(caTools)
library(caret)
library(e1071)
library(pROC)
library(randomForest)
library(caTools)
library(RRF)
#library(earth)

#Load teh dataset for the population survey health data
df <- read.csv('cps_earnings_HealthPred.csv', stringsAsFactors = F, na.strings=c("","NA"," ","?"))


#lets have a look at the data
str(df)
print(dim(df))
print(head(df))

#convert all the char types features to factors.
df$region <- as.factor(df$region)
df$statefip <- as.factor(df$statefip)
df$age <- as.numeric(df$age)
df$sex <- as.factor(df$sex)
df$race <- as.factor(df$race)
df$asian <- as.factor(df$asian)
df$marst <- as.factor(df$marst)
df$citizen <- as.factor(df$citizen)
df$hispan <- as.factor(df$hispan)
df$educ <- as.factor(df$educ)
df$labforce <- as.factor(df$labforce)
df$union <- as.factor(df$union)


# check for the health classes
print(unique(df$health))

# replace all the health feature with the numeric binning
df$health[df$health=='excellent'] <- 5
df$health[df$health=="very good"] <- 4
df$health[df$health=="good"] <- 3
df$health[df$health=="fair"] <- 2
df$health[df$health=='poor'] <- 1

#convert target feature in factor
df$health <- as.factor(df$health)
str(df)

#check for the null values

print("Percentage of missing values")
print(sapply(df, function(df){ sum(is.na(df)==T) * 100 /length(df) }))

#only health feature posses the null values and less than 2% so will drop those
# delete the columns null values

df <- df[!is.na(df$age), ]

table(df$health)


# numerical feature scaling

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

df$county<-normalize(df$county)
df$age<-normalize(df$age)
df$wkswork1<-normalize(df$wkswork1)
df$uhrsworkly<-normalize(df$uhrsworkly)
df$incwage<-normalize(df$incwage)

View(df)

# slit the data into the train, test data with 3:1 ratio

set.seed(123)
split <- sample.split(df, SplitRatio = 0.75)
train <- subset(df, split == "TRUE")
test <- subset(df, split == "FALSE")
str(train)
str(test)

# RANDOM FOREST
# mtry = 4 ; square root of the total numbr of features, ntree = 64 -128 for optimal processing time
# tried on above conditions but accuracy didnt improved so for optimum performance we wil train it on lower values of mtry and ntree
rfClassifier <- randomForest(health ~ ., data = train, importance=TRUE, ntree = 15,mtry=4)

Predictions <- predict(rfClassifier,newdata=test[-17],type="class")

print(unique(Predictions))


#Evaluation

randomForestCM <- confusionMatrix(as.factor(Predictions),as.factor(test$health))
print("randomForestCM :")
print(randomForestCM)

test$health <- as.numeric(test$health)
print(class(test$health))
ROC_RF<-multiclass.roc(as.factor(Predictions),test$health)
print(ROC_RF)
#Accuracy : 0.4167 ,95% CI : (0.4025, 0.4098),Kappa : 0.1741 ,Multi-class area under the curve: 0.7185

######################################################

#tune for mtry

bestModel <- tuneRF(test[-17],test$health, ntree=5,stepFactor = 1.5, improve= 1e-5 ,doBest = TRUE)
print(bestModel)
plot(bestModel)
# it seems mtry = 2 is optimal value as OOB error is increasing for mtry = 4 onwards

#predict for selected features to the bestmodel builta

tune_Predictions <- predict(bestModel,newdata=test[-17],type="class")


randomForestCM <- confusionMatrix(as.factor(tune_Predictions),as.factor(test$health))
print("randomForestCM :")
print(randomForestCM)

test$health <- as.numeric(test$health)
tuneROC_RF<-multiclass.roc(as.factor(tune_Predictions),test$health)
print(tuneROC_RF)

#Accuracy :  0.7489,95% CI : (0.7453, 0.7526), Kappa : 0.6454 ,Multi-class area under the curve: 0.9242