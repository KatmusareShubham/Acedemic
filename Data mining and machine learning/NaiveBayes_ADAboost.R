#3
library(dplyr)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(caTools)
library(caret)
library(e1071)
library(pROC)
library(naivebayes)
library(adabag)
library(klaR)

# load the bank customer survey data
df <- read.csv('DMML1/bankFullSurvey.csv', stringsAsFactors = F, na.strings=c("","NA"," ","?"))

# lets have a look ate the data
head(df)
dim(df)
str(df)


#convert all the char types features to factors.
df$job <- as.factor(df$job)
df$marital <- as.factor(df$marital)
df$education <- as.factor(df$education)
df$default <- as.factor(df$default)
df$housing <- as.factor(df$housing)
df$loan <- as.factor(df$loan)
df$contact <- as.factor(df$contact)
df$month <- as.factor(df$month)
df$day_of_week <- as.factor(df$day_of_week)
df$poutcome <- as.factor(df$poutcome)
#df$emp.var.rate <- integer(df$emp.var.rate)
#df$cons.price.idx <- integer(df$cons.price.idx)
df$cons.conf.idx <- as.factor(df$cons.conf.idx)
df$euribor3m <- as.factor(df$euribor3m)
df$nr.employed <- as.factor(df$nr.employed)

#Map 0 to 1 the target vaiable "y"
print(unique(df$y))
df$y<-ifelse(df$y=='yes',1,0)

# for classification , target variable should be factor
df$y <- as.factor(df$y)
str(df)


# scaling for  the numeric columns
# numerical feature scaling

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

df$age<-normalize(df$age)
df$duration<-normalize(df$duration)
df$pdays<-normalize(df$pdays)
df$cons.price.idx<-normalize(df$cons.price.idx)

#Check for null values

print("Percentage of missing values")
print(sapply(df, function(df){ sum(is.na(df)==T) * 100 /length(df) }))

# no nul values in the data

table(df$y)

# Split the data into train and test
set.seed(3456)
split <- sample.split(df, SplitRatio = 0.75)
train <- subset(df, split == "TRUE")
test <- subset(df, split == "FALSE")
str(train)
str(test)

#Df target classes ditribution
table(df$y) %>% prop.table()

# check if target is balanced
table(train$y) %>% prop.table()
table(test$y) %>% prop.table()

#With naïve Bayes, we assume that the predictor variables are conditionally independent of one another given the response value. This is an extremely strong assumption. We can see quickly that our attrition data violates this as we have several moderately to strongly correlated variables.

train %>%
  filter(y == '1') %>%
  select_if(is.numeric) %>%
  cor() %>%
  corrplot::corrplot()

#check for the numerical columns properties
summary(df)

features <- setdiff(names(train), "y")
xdata <- train[, features]
ydata <- train$y



# Naive Baye's classifier implementaion

train_control <- trainControl(
  method = "cv",
  number = 10
)

NBclassifier <- train( x = xdata, y = ydata, trControl = train_control,laplace = 1, method= 'nb')
Predictions<- predict(NBclassifier,newdata=test[-21])

table(Predictions)
table(test$y)

NaievbaysCM = confusionMatrix(as.factor(Predictions),as.factor(test$y))
print(NaievbaysCM)
print(ROC_NB)

#test$y <- as.numeric(test$y)
#print(class(test$y))
ROC_NB<-roc(Predictions,as.numeric(test$y))
print(ROC_NB)
plot(ROC_NB, col = "#fd634b", family = "sans", cex = 2, main = "Naive BAYE ROC Curve - AUC = 0.7127")
#Accuracy : 0.8833,95% CI : (0.864, 0.8763), Kappa : 0.4515 , Sensitivity : 0.9244 ,Specificity : 0.5580,AUC- 0.7127


#Performace tuning for NV

search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = 0:5,
  adjust = seq(0, 5, by = 1)
)

tunedNB<- NaiveBayes(x = xdata,y = ydata,
                trControl = train_control,
                tuneGrid = search_grid,
                preProc = c("BoxCox", "center", "scale", "pca")
)
print(tunedNB)


tunedPredictions<- predict(tunedNB,newdata=test[-21])
TunedNaievbaysCM = confusionMatrix(as.factor(tunedPredictions),as.factor(test$y))
print(TunedNaievbaysCM)
tunedROC_NB<-roc(tunedPredictions,as.numeric(test$y))
print(tunedROC_NB)
plot(tunedROC_NB, col = "#fd634b", family = "sans", cex = 2, main = "NAive-Baye's ROC-AUC :0.7065")
###Accuracy : 0.8794, 95% CI : (0.8733, 0.8852),Kappa : 0.4469,Sensitivity : 0.9182,Specificity : 0.5716Area under the curve: 0.7065     



#Adaboost implementation

model = boosting(y~., data=train, boos=TRUE)
Pred = predict(model, test[-21])$class

print(Pred)
print(factor(Pred))

AdaCM = confusionMatrix(as.factor(Pred),as.factor(test$y))

ROC_ADA<-roc(Pred,as.numeric(test$y))
print(AdaCM)
print(ROC_ADA)
plot(ROC_ADA, col = "#fd634b", family = "sans", cex = 2, main = "Adaboost ROC  - AUC = 0.7713")
#Accuracy : 0.908, 95% CI : (0.9037, 0.9142), Kappa : 0.514, Sensitivity : 0.96, Specificity : 0.5337,AUC = 0.7713

##############################################
# feature selection for adaboost

set.seed(7)

model <- train(y ~ ., data=train, method="rpart", )
impfetures <- varImp(model)
print(impfetures)

# seems like very few columns contribute to the target varibale so will filter it out and train the model
ft_train <- subset(train, select =  c(age,pdays,poutcome,duration,emp.var.rate,previous,cons.conf.idx,nr.employed,y))
ft_test <- subset(test, select = c(age,pdays,poutcome,duration,emp.var.rate,previous,cons.conf.idx,nr.employed,y))

print(ft_train)
# trained the model on the selected features and compare the performance

ftmodel = boosting(y~., data=ft_train, boos=TRUE)
Pred = predict(ftmodel, ft_test[-9])$class

ftAdaCM = confusionMatrix(as.factor(Pred),as.factor(ft_test$y))
ROC_ADA<-roc(Pred,as.numeric(ft_test$y))
print(ftAdaCM)
print(ROC_ADA)
plot(ROC_ADA, col = "#fd634b", family = "sans", cex = 2, main = "Adaboost ROC  - AUC = 0.7862")

#Accuracy : 0.9126,Kappa : 0.5284, Sensitivity : 0.9607,Specificity : 0.5315, Area under the curve: 0.7862
