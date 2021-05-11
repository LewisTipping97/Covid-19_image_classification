#Importing test data
# Setting the wd and seed, and importing relevant libraries
setwd("~/Documents/Coding/CFAS420/Data")
library(magick) 
library(caret)
library(vcd)
library(xgboost)
library(ISLR)
library(skimr)
library(rpart)
library(rattle)
library(dplyr)
library(bst)
set.seed(1)


#COVID DATA
files = list.files(path = "COVID" , full.names=TRUE)
img <- image_read (files[1]) 
mat <- as.numeric(image_data(img))[,,1]
vec <- as.vector(mat)

#A loop was created which appended the imagedata to the existing dataframe
# this was then repeated for all three image folders: Covid, Non-Covid and the test data.

for (i in 2:length(files)){
  img <- image_read (files[i]) #read first image in folder
  mat <- as.numeric(image_data(img))[,,1]
  vec1 <- as.vector(mat)
  vec <- rbind(vec, vec1)
}
covid_data <- as.data.frame(vec)
outcome <- c(rep(1, 250))
covid_data1 <- cbind(covid_data, outcome)

#NON COVID DATA
files1 = list.files(path = "NON_COVID" , full.names=TRUE)
img1 <- image_read (files1[1]) # r e ad f i r s t image i n f o l d e r
mat1 <- as.numeric(image_data(img1))[,,1]
vec2 <- as.vector(mat1) # Conv e r t to v e c t o r


for (i in 1:length(files1)){
  img1 <- image_read (files1[i]) #read first image in folder
  mat <- as.numeric(image_data(img1))[,,1]
  vec1 <- as.vector(mat)
  covid_data <- rbind(covid_data, vec1)
}
#TEST DATA
files2 = list.files(path = "TEST" , full.names=TRUE)
img2 <- image_read (files2[1]) 
mat2 <- as.numeric(image_data(img2))[,,1]
vec3 <- as.vector(mat2) 
mat3 <- matrix(vec3,200,200) 
image(mat3) 

for (i in 1:length(files2)){
  img1 <- image_read(files2[i]) #read first image in folder
  mat <- as.numeric(image_data(img1))[,,1]
  vec1 <- as.vector(mat)
  covid_data <- rbind(covid_data, vec1)
}
#write.csv(covid_data, 'covid_data.csv')

# I saved the full data frame as a new csv file to stop myself having to run the
# code everytime I opened the project
covid_data <- read.csv('covid_data.csv')
covid_data<- covid_data[,2:40001]
#covid_data is now a dataset with all the avaliable data, no labels have been assigned
#the first 250 rows are covid = yes, 251-500 rows covid = no, last 98 rows is the test data
#PCA on full dataset
pca_data <- prcomp(covid_data, scale = TRUE, rank =100)
summary(pca_data)


#The percent that each PC explains as a list, plotted
pcent_var_expln <- 100*pca_data$sdev^2/sum(pca_data$sdev^2)
plot(pcent_var_expln, type='o', xlab = 'Principal Component',
     ylab = 'Percentage of variance explained',
     main = 'Percentage of variance explained for all PC\'s')
#The first 100 PCs explain ~81% of the data
sum(pcent_var_expln[1:100])

#A new df which represents the reduced data
reduced_data <- as.data.frame(pca_data$x)

#Can also load this file in independently if needed
#write.csv(reduced_data, 'reduced_data.csv')

#The outcome variables as a list
outcomes <- c(rep(1, 250), rep(0, 250))
#Adding the outcome to the df
reduced_data_w_outcome <- cbind(reduced_data[1:500,], outcomes)
#write.csv(reduced_data_w_outcome, 'reduced_data_w_outcomes.csv')

# Turning the data into a factor type                       
reduced_data_w_outcome$outcomes <- as.factor(reduced_data_w_outcome$outcomes)
levels(reduced_data_w_outcome$outcomes)[levels(reduced_data_w_outcome$outcomes)== '0'] <- 'Not_Covid'
levels(reduced_data_w_outcome$outcomes)[levels(reduced_data_w_outcome$outcomes)== '1'] <- 'Covid'


#Plotting PC1 against PC2 to see if theres any clear viusal patterns
library(ggplot2)
ggplot(data = reduced_data_w_outcome, aes(x=PC1, y=PC2, color=outcomes)) +geom_point() +
  labs(title = 'The relationship between PC1 and PC2')
#CLASSIFICATION STARTS HERE - XGB

#training/test split
train_indicies <- createDataPartition(reduced_data_w_outcome$outcomes,
                                      p=0.8,
                                      list = FALSE,
                                      times = 1)
train_df <- reduced_data_w_outcome[train_indicies,]
test_df <- reduced_data_w_outcome[-train_indicies,]
#XGB model building
trctrl <- trainControl(method = "cv", number = 5)

#The parameters being tuned are eta and number of rounds, so they are in list form
tune_grid <- expand.grid(nrounds = c(1:200),
                         max_depth = 2,
                         eta = c(seq(from =0.01, to = 0.3, by = 0.01)),
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)
#Training the model, the outcome is being used as the dependent variable
xgb_fit <- caret::train(outcomes ~., data = train_df, method = "xgbTree",
                       trControl=trctrl,
                       tuneGrid = tune_grid,
                       tuneLength = 10)


plot(xgb_fit, main = 'XGBoost Tuning') #plots the resulting graph (all the shrinkages and iterations)
xgb_fit$bestTune #best model (to maximise accuracy)
#predicting model on test data
predictions <- predict(xgb_fit, newdata = test_df[1:100], type = 'prob')
predicted_xgb <- as.factor(predictions[,2]> 0.5) #Default threshold of 0.5
levels(predicted_xgb) <- c("Not_Covid", "Covid") #renaming factors

#How the best model performed
max(xgb_fit$results$Accuracy)


#Confusion matrix with the test predictions and test results
confusionMatrix(predicted_xgb, test_df$outcomes)


library(pROC)
my_roc <- roc(as.integer(test_df$outcomes)-1, unlist(predictions[2]))

my_roc$sensitivities #different possible sensitivities - only these values can be picked as potential values

# Seeing what threshold is needed for a 96% sensitivity
coords(my_roc, x=0.96, input = 'sensitivity', transpose = T, ret = c('sensitivity','specificity',                                                                     'accuracy', 'threshold'))
#0.2807614 Threshold^

coords(my_roc, x='best', transpose = T, ret = c('sensitivity','specificity',
                                                'accuracy', 'threshold'))
#0.4973199 Threshold^

#Plotting the ROC curve with the two points labelled
plot(my_roc, col = 'red', main = 'ROC curve for XGBoost') #roc curve
points(x=0.92, y=0.9, pch = 19)
points(x=0.72, y=0.96, pch = 19, col = 'blue')
legend(0.1, 0.3, legend = c('Highest Accuracy', '96% sensitvity'), col = c('black', 'blue'), 
       pch =19, cex=0.8, title = 'Outcome')




#LOGISTIC REGRESSION

library(glmnet)
library(LogicReg)
library(stepPlr)

#The model needed to be in matrix form for logistic reg
x <- model.matrix(outcomes~., train_df)[,-1]
# Convert the outcome (class) to a numerical variable
y <- ifelse(train_df$outcomes == "Covid", 1, 0)

#Finding optimal lambda
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv.lasso)
title('Grid search for optimal Lambda', line = 3)

#Tuning grid created with 1SE lambda, alpha is 1 for Lasso
tuningGrid_log <- expand.grid(lambda =cv.lasso$lambda.1se , alpha = 1)
log_model <- caret::train(outcomes ~., data = train_df, method = "glmnet",
                          tuneGrid = tuningGrid_log,
                          trControl=trctrl)

#Predicting on test data
log_pred <- predict(log_model, newdata = test_df, type = 'prob')

#ROC curve with the log model and test data
my_roc_log <- roc(as.integer(test_df$outcomes)-1, unlist(log_pred[2]))

my_roc_log$sensitivities

#finding the best threshold and its corresponding results
coords(my_roc_log, x='best', transpose = T, ret = c('sensitivity','specificity',
                                                    'accuracy', 'threshold'))

#Focusing on a higher sensitivity
coords(my_roc_log, x=0.98, input = 'sensitivity', transpose = T, ret = c('sensitivity','specificity',
                                                                         'accuracy', 'threshold'))



#coef(cv.lasso, cv.lasso$lambda.1se) 63 coefficients present
cv.lasso$lambda.1se

#Finding how many coefs are 0 for minimum lambda 36
sum(coef(cv.lasso, cv.lasso$lambda.min)==0)

#finding how many coefs are 0 for the 1se lambda 54
sum(coef(cv.lasso, cv.lasso$lambda.1se)==0)


my_roc_log$sensitivities

#This was commented as the final graph didnt involve these points
plot(my_roc_log, main = 'ROC curve comparison', col = 'blue')
#points(x=0.54, y=0.98, pch = 19)
#points(x=0.72, y= 0.88, pch = 19, col = 'red')
#legend(0.1, 0.3, legend = c('Highest Accuracy', '98% sensitvity'), col = c('red', 'black'), 
#       pch =19, cex=0.8, title = 'Outcome')

plot(my_roc, add = T, col = 'red') 
legend(0.2, 0.3, legend = c('Logistic Regression', 'XGBoost'), col = c('blue', 'red'), 
      lty = 1:1 ,cex=0.8, title = 'Model')


#Creating a Covid/not covid prediction
predicted_log <- as.factor(log_pred[,2]> 0.5)
levels(predicted_log) <- c("Not_Covid", "Covid") #renaming factors



#TESTING DATA on the unseen data (the final 98 observations)
to_test <- reduced_data[501:598,]
subject_id <- c(1:98)

#predicting on unseen data, returns a probability that patient is covid positive
final_prediction <- predict(xgb_fit, to_test, type = 'prob')

#Setting threshold such that it will give me the highest accuracy
predicted_class <- as.numeric(final_prediction[,2]> 0.4973199)
test_predictions <- cbind(subject_id, predicted_class)

#write.csv(test_predictions, 'student_id_predictions.csv')





