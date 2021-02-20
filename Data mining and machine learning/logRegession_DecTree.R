#1
library(dplyr)
library(sqldf)
library(ggplot2)
library(reshape2)
library(gridExtra)
library('caTools')
library(caret)
library(e1071)
library(pROC)
library(mlbench)
library(party)

# import the dataset from the local machine
df <- read.csv('DMML1/Fin_survey_data_.csv', stringsAsFactors = F, na.strings=c("","NA"," ","?"))

#head(df)

#Index column delete 
df <- df[-c(1)] 


#convert all the char types features to factors.
df$workclass <- as.factor(df$workclass)
df$education <- as.factor(df$education)
df$marital.status <- as.factor(df$marital.status)
df$occupation <- as.factor(df$occupation)
df$relationship <- as.factor(df$relationship)
df$race <- as.factor(df$race)
df$gender <- as.factor(df$gender)
df$native.country <- as.factor(df$native.country)


#Map 0 to 1 the target feature -- Income
df$income<-ifelse(df$income=='>50K',1,0)

#str(df)

# check for the the unique values confirmation (further ---factors must be two "0","1")
#print(unique(df$income))


#print("Percentage of missing values")
print(sapply(df, function(df){ sum(is.na(df)==T) * 100 /length(df) }))

# delete the null values as those are less than 5 % of all teh data. 

df <- df[!is.na(df$workclass), ]
df <- df[!is.na(df$occupation), ]

df$native.country[is.na(df$native.country)] <- names(which.max(table(df$native.country,useNA="no")))

#print("Percentage of missing values")
#print(sapply(df, function(df){ sum(is.na(df)==T) * 100 /length(df) }))

# check for the target class factor distribution
#table(df$income)

# set the seed for random state selection
set.seed(123)

# split the data 3:1 (train:test ratio)
split <- sample.split(df, SplitRatio = 0.75)
train <- subset(df, split == "TRUE")
test <- subset(df, split == "FALSE")
str(train)
str(test)

# Implementation of the Logistic regression Classifier

LRclassifier <-glm(income ~.,family="binomial",data=train)
Predictions<- predict(LRclassifier,newdata=test[-15],type = 'response')

#print(Predictions)
Pred<- ifelse(Predictions>0.5,1,0)
Pred_y <- factor(Pred,levels=c(0,1))
testIncome <- factor(test$income,levels = c(0,1))

#table(Pred)
#table(testIncome)

logisticREgressionCM <- confusionMatrix(as.factor(Pred),as.factor(testIncome))
#print("logisticREgressionCM :")
print(logisticREgressionCM)

#Accuracy : 0.846, 95% CI : (0.8421, 0.8549),  Kappa : 0.5757, Sensitivity : 0.9270, AUC = 0.8091

ROC_LR<-roc(Pred,test$income)
print(ROC_LR)
plot(ROC_LR, col = "#fd634b", family = "sans", cex = 2, main = "Featured logistic regression tree- ROC Curve - AUC = 0.8091")

###############################################################


#DECISON TREE IMPLEMENTATION

library(rpart)
library(rpart.plot)
suppressMessages(library(rpart))
suppressMessages(library(rpart.plot))

DTclassifier <- rpart(income ~., data = train, method = "class")
summary(finalDct)

Income <- predict(DTclassifier,newdata=test[-15],type="class")
table(predictedincome)
table(test$income)

DecisionTreeCM <- confusionMatrix(as.factor(Income),as.factor(test$income))
print(DecisionTreeCM)

ROC_DTT<-roc(Income,test$income)
print(ROC_DTT)
plot(ROC_DTT, col = "#fd634b", family = "sans", cex = 2, main = "Featured decision tree- ROC Curve - AUC = 0.7987")

# Accuracy : 0.7771 ,95% CI : (0.7697, 0.7845),Kappa : 0.3515 , Sensitivity : 0.8973  , AUC = 0.7987
################################################################
# feature selection using random forest

set.seed(7)

model <- cforest(income ~ ., data=train, control = cforest_unbiased(mtry= 5,ntree =10))
ImpFetures <- varImp(model)
print(ImpFetures*1000)

# seems like very few columns contribute to the target variable so will filter it out and train the model
ft_train <- subset(train, select = - c(educational.num,capital.gain,relationship,hours.per.week,workclass,marital.status,age,occupation))
ft_test <- subset(test, select = - c(educational.num,capital.gain,relationship,hours.per.week,workclass,marital.status,age,occupation))

################################################################

#implement the Logistic regession with featured variables

LRclassifier <-glm(income ~.,family="binomial",data=ft_train)
Predictions<- predict(LRclassifier,newdata=ft_test[-7],type = 'response')

#print(Predictions)
ftPred<- ifelse(Predictions>0.5,1,0)
ftPred_y <- factor(Pred,levels=c(0,1))
fttestIncome <- factor(fttest$income,levels = c(0,1))

#table(Pred)
#table(testIncome)

logisticREgressionCM <- confusionMatrix(as.factor(ftPred),as.factor(fttestIncome))
#print("logisticREgressionCM :")
print(logisticREgressionCM)
ROC_LR<-roc(ftPred,ft_test$income)
print(ROC_LR)
#plot(ROC_LR, col = "#fd634b", family = "sans", cex = 2, main = "Logistic Regression- ROC Curve - AUC = .8091")
#Accuracy : 0.8486 ,95% CI : (0.8279, 0.8411),Kappa : 0.5767 ,Sensitivity : 0.9270 ,Area under the curve: .8091

####################################################################

#Decision tree implementation with the selected features

DTclassifier <- rpart(income ~., data = ft_train, method = "class",maxdepth = 5)
summary(DTclassifier)

Income <- predict(DTclassifier,newdata=ft_test[-7],type="class")
table(Income)
table(ft_test$income)

DecisionTreeCM <- confusionMatrix(as.factor(Income),as.factor(ft_test$income))
print(DecisionTreeCM)
rpart.plot(DTclassifier)

ROC_DT<-roc(ftPred,ft_test$income)
print(ROC_DT)
#plot(ROC_DT, col = "#fd634b", family = "sans", cex = 2, main = "Featured decision tree- ROC Curve - AUC = 0.8036")

# Accuracy : 0.8383 ,95% CI : (0.8317, 0.8448),Kappa : 0.5291 , Sensitivity : 0.9384 , AUC- 0.8036

